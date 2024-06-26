import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math
from numpy.lib import stride_tricks
from utils import overlap_and_add, device
from torch.autograd import Variable

EPS = 1e-8

def smooth_filter(tensor):
    kernel_size = 11
    padding = kernel_size // 2
    tensor = F.pad(tensor, (padding, padding), mode='reflect')
    tensor = F.avg_pool1d(tensor, kernel_size=kernel_size, stride=1)
    return tensor

class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [B, N, L]
        return mixture_w

    
class Decoder(nn.Module):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask=0):
        """
        Args:
            mixture_w: [B, E, L]
            est_mask: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3) # [B, C, L, E]
        # S = DV
        est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W//2) # B x C x T
        return est_source

class RecNet(nn.Module):
    def __init__(self, nsplit, dim, k_size):
        super(RecNet, self).__init__()
        # Hyper-parameter
        self.nsplit = nsplit
        self.T = 8000
        # Components
        self.enc = nn.Linear(self.nsplit * self.T, dim)
        self.linear = nn.Linear(dim, dim)
        self.dec = nn.Linear(dim, self.T)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [B, nsplit, T]

        Returns:
            y: [B, 1, T]
        """
        x = x.view(-1, 1, self.nsplit * self.T)
        x = self.enc(x)
        #x = self.norm2(self.act(x))
        x = self.linear(x)
        #x = self.norm2(self.act(x))
        x = self.dec(x)
        #x = self.norm(self.act(x))
        y = torch.unsqueeze(x, 1)
        #y = x.permute(0, 2, 1)
        #y = torch.sum(x, dim=1)
        return y
    
class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

class SingleTransformer(nn.Module):
    """
    Container module for a single Transformer layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        num_heads: int, number of attention heads. Default is 4.
    """

    def __init__(self, input_size, hidden_size, dropout=0, num_heads=4):
        super(SingleTransformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # self-attention layer
        self.attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)

        # feed-forward layer
        self.ffn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )

        # layer normalization
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input

        # transpose the input for attention
        output = output.transpose(0, 1)

        # self-attention with residual connection and layer normalization
        attn_output, _ = self.attention(output, output, output)
        output = self.ln1(output + attn_output)

        # feed-forward with residual connection and layer normalization
        ffn_output = self.ffn(output)
        output = self.ln2(output + ffn_output)

        # transpose the output back
        output = output.transpose(0, 1)

        return output

class Sepformer(nn.Module):
    """
    Deep duaL-path Transformer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked Transformer layers. Default is 1.
        num_heads: int, number of attention heads. Default is 4.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, num_heads=4):
        super(Sepformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path Transformer
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(SingleTransformer(input_size, hidden_size, dropout, num_heads))  # intra-segment Transformer is always noncausal
            self.col_trans.append(SingleTransformer(input_size, hidden_size, dropout, num_heads))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                    )

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_trans)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_trans[i](row_input)  # B*dim2, dim1, N
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_trans[i](col_input)  # B*dim1, dim2, N
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                             2).contiguous()  # B, N, dim1, dim2
            output = output + col_output

        output = self.output(output) # B,output_size,dim1,dim2

        return output
    
# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size, output_size, 1)
                                    )

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                             2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output) # B, output_size, dim1, dim2

        return output


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_split=2,
                 layer=4, segment_size=100, bidirectional=True, rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_split = num_split

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPRNN model
        self.DPRNN = DPRNN(rnn_type, self.feature_dim, self.hidden_dim,
                                   self.feature_dim * self.num_split,
                                   num_layers=layer, bidirectional=bidirectional)

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass

# base module for sepformer
class Sepformer_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_split=2,
                 layer=10, segment_size=100):
        super(Sepformer_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_split = num_split

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPRNN model
        self.Sepformer = Sepformer(self.feature_dim, self.hidden_dim,
                                   self.feature_dim * self.num_split,
                                   num_layers=layer)

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass

# DPRNN for beamforming filter estimation
class BF_module2(Sepformer_base):
    def __init__(self, *args, **kwargs):
        super(BF_module2, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, input):
        #input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input) # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K: L is the segment_size
        #print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPRNN
        output = self.Sepformer(enc_segments).view(batch_size * self.num_split, self.feature_dim, self.segment_size,
                                                   -1)  # B*nsplit, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*nsplit, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nsplit, K, T
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_split, -1,
                                                                self.feature_dim)  # B, nsplit, T, N

        return bf_filter
    
# Sepformer for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, input):
        #input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input) # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K: L is the segment_size
        #print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPRNN
        output = self.DPRNN(enc_segments).view(batch_size * self.num_split, self.feature_dim, self.segment_size,
                                                   -1)  # B*nsplit, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*nsplit, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nsplit, K, T
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_split, -1,
                                                                self.feature_dim)  # B, nsplit, T, N

        return bf_filter
    
# base module for AE
class AE_base(nn.Module):
    def __init__(self, enc_dim, win_len=2):
        super(AE_base, self).__init__()
        self.encoder = Encoder(win_len, enc_dim)
        self.decoder = Decoder(enc_dim, win_len)

    def forward(self, input):
        mixture_w = self.encoder(input)
        est_source = self.decoder(mixture_w)
        return est_source



# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250,
                 nsplit=2, win_len=2):
        super(FaSNet_base, self).__init__()
    

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_split = nsplit
        self.eps = 1e-8

        # waveform encoder
        #self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim) # [B T]-->[B N L]
       
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                self.num_split, self.layer, self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        #input = input.to(device)
        B, _ = input.size()
        # mixture, rest = self.pad_input(input, self.window)
        #print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L

        score_ = self.enc_LN(mixture_w) # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        score_ = self.separator(score_)  # B, nsplit, T, N
        #print('score_.shape {}'.format(score_.shape))
        score_ = score_.view(B*self.num_split, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nsplit, N, T
        #print('score_.shape {}'.format(score_.shape))
        score = self.mask_conv1x1(score_)  # [B*nsplit, N, L] -> [B*nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        score = score.view(B, self.num_split, self.enc_dim, -1)  # [B*nsplit, E, L] -> [B, nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        est_mask = F.relu(score)
        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nsplit, E, L]--> [B, nsplit, T]

        #     est_source = est_source[:, :, :-rest]

        return est_source
        
# base module for FaSNet
class FaSNet_mod1(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250,
                 nsplit=2, win_len=2):
        super(FaSNet_mod1, self).__init__()
    

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_split = nsplit
        self.eps = 1e-8

        # waveform encoder
        #self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim) # [B T]-->[B N L]
       
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                self.num_split, self.layer, self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.rdecoder = RecNet(nsplit, 256, 16)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        #input = input.to(device)
        B, _ = input.size()
        # mixture, rest = self.pad_input(input, self.window)
        #print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L

        score_ = self.enc_LN(mixture_w) # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        score_ = self.separator(score_)  # B, nsplit, T, N
        #print('score_.shape {}'.format(score_.shape))
        score_ = score_.view(B*self.num_split, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nsplit, N, T
        #print('score_.shape {}'.format(score_.shape))
        score = self.mask_conv1x1(score_)  # [B*nsplit, N, L] -> [B*nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        score = score.view(B, self.num_split, self.enc_dim, -1)  # [B*nsplit, E, L] -> [B, nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        est_mask = F.relu(score)
        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nsplit, E, L]--> [B, nsplit, T]
        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]
        rec_source = self.rdecoder(est_source)

        return est_source,rec_source
# base module for FaSNet
class FaSNet_base2(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250,
                 nsplit=2, win_len=2):
        super(FaSNet_base2, self).__init__()
    

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_split = nsplit
        self.eps = 1e-8

        # waveform encoder
        #self.encoder = nn.Conv1d(1, self.enc_dim, self.feature_dim, bias=False)
        self.encoder = Encoder(win_len, enc_dim) # [B T]-->[B N L]
       
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8) # [B N L]-->[B N L]
        self.separator = BF_module2(self.enc_dim, self.feature_dim, self.hidden_dim,
                                self.num_split, self.layer, self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        #input = input.to(device)
        B, _ = input.size()
        # mixture, rest = self.pad_input(input, self.window)
        #print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L

        score_ = self.enc_LN(mixture_w) # B, E, L
        #print('mixture_w.shape {}'.format(mixture_w.shape))
        score_ = self.separator(score_)  # B, nsplit, T, N
        #print('score_.shape {}'.format(score_.shape))
        score_ = score_.view(B*self.num_split, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nsplit, N, T
        #print('score_.shape {}'.format(score_.shape))
        score = self.mask_conv1x1(score_)  # [B*nsplit, N, L] -> [B*nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        score = score.view(B, self.num_split, self.enc_dim, -1)  # [B*nsplit, E, L] -> [B, nsplit, E, L]
        #print('score.shape {}'.format(score.shape))
        est_mask = F.relu(score)
        est_source = self.decoder(mixture_w, est_mask) # [B, E, L] + [B, nsplit, E, L]--> [B, nsplit, T]
        # if rest > 0:
        #     est_source = est_source[:, :, :-rest]

        return est_source
