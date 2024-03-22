import math
import random
from trend import TrendSequenceGenerator
import scipy.signal as signal

def quantize(seq, levels):
    """
    A function to quantize a sequence with a given number of levels.

    args:
        seq: a list of numbers, the sequence to be quantized.
        levels: an int, the number of levels to quantize the sequence.

    returns:
        a list of numbers, the quantized sequence.
    """

    # find the minimum and maximum values in the sequence
    min_val = min(seq)
    max_val = max(seq)

    # calculate the step size for each level
    step = (max_val - min_val) / levels

    # initialize an empty list for the quantized sequence
    quantized_seq = []

    # loop through each value in the sequence
    for val in seq:
        # calculate the level index for this value
        level = int((val - min_val) / step)
        # clip the level index to be within [0, levels - 1]
        level = max(0, min(level, levels - 1))
        # calculate the quantized value for this level
        quantized_val = min_val + level * step
        # append it to the quantized sequence
        quantized_seq.append(quantized_val)

    return quantized_seq

class PeriodicSequenceGenerator:
    """
    A class to generate a periodic sequence with a given amplitude and period.

    args:
        amplitude: float, the amplitude parameter of the sequence. Default is 1.
        period: int, the period parameter of the sequence. Default is 10.
        length: int, the length of the sequence. Default is 100.
        type: string, the type of the sequence. Choose from 'sine', 'square', 'randsine', 'randsquare', 'randseason'. Default is 'sine'.
    """

    def __init__(self, amplitude=1, period=10, length=100, type='sine', wavnum=1, seq=[], delta_s=1, shake_factor=0.01, limit_type='amp_unlimited',levels=5,flex_factor=0,smooth = 0):
        self.amplitude = amplitude
        self.period = period
        self.length = length
        self.type = type
        self.wavnum = wavnum
        self.seq = seq
        self.delta_s = delta_s
        self.shake_factor = shake_factor
        self.limit_type = limit_type
        self.levels = levels
        self.flex_factor = flex_factor 
        self.smooth = smooth 

 
    def generate(self):
        # initialize an empty sequence
        seq = []

        # generate the sequence according to the type
        if self.type == 'sine':
            # generate a sine wave with the given amplitude and period
            for i in range(self.length):
                # calculate the angle in radians
                angle = 2 * math.pi * i / self.period
                # calculate the value of the sine wave at this angle
                value = self.amplitude * math.sin(angle)
                # append it to the sequence
                seq.append(value)

        elif self.type == 'square':
            # generate a square wave with the given amplitude and period
            for i in range(self.length):
                # calculate the phase of the square wave
                phase = i % self.period
                # determine the value of the square wave at this phase
                if phase < self.period / 2:
                    value = self.amplitude # high state
                else:
                    value = -self.amplitude # low state
                # append it to the sequence
                seq.append(value)

        elif self.type == 'randsine':
            # generate a random sine wave
            amplitude_list = []
            phase_list = []
            freq_list = []
            for i in range(self.length):
                # generate random amplitude
                amp = random.uniform(0, self.amplitude)
                amplitude_list.append(amp)
                # generate random phase
                phase = random.uniform(0, 2*math.pi)
                phase_list.append(phase)
                # generate random frequency
                freq_limit = 2 * math.pi / self.period
                freq = random.uniform(freq_limit/10, freq_limit)
                freq_list.append(freq)

            for i in range(self.length):
                # initialize the value
                value = 0
                for j  in range(self.wavnum):
                    angle = i * freq_list[j] + phase_list[j]
                    value += amplitude_list[j] * math.sin(angle)
                    # append it to the sequence
                seq.append(value)
        
        elif self.type == 'randsquare':
            # generate a random square wave
            amplitude_list = []
            phase_list = []
            period_list = []
            for i in range(self.length):
                # generate random amplitude
                amp = random.uniform(0, self.amplitude)
                amplitude_list.append(amp)
                # generate random period
                period = random.randint(int(self.period/10), self.period)
                period_list.append(period)
                # generate random phase
                phase = random.randint(0, period)
                phase_list.append(phase)

            for i in range(self.length):
                # initialize the value
                value = 0
                for j  in range(self.wavnum):
                    # calculate the phase of the square wave
                    phase_ = (i +  phase_list[j]) % period_list[j]
                    # determine the value of the square wave at this phase
                    if phase_ < period_list[j] / 2:
                        value += amplitude_list[j] # high state
                    else:
                        value += -amplitude_list[j] # low state
                    # append it to the sequence
                    
                seq.append(value)

        elif self.type == 'randseason':
            # generate a normal seasonal sequence
            if len(self.seq) > 0:
                # sequence is set by hand
                seq_ = self.seq
                period = self.period
            else:
                # sequence is generated automaticaly
                period = random.randint(int(self.period/3), self.period)
                t = TrendSequenceGenerator(amplitude=self.amplitude, delta_s=self.delta_s, length=period,type=self.limit_type)
                seq_ = t.generate()
            # generate
            
            phase = random.randint(0, period)
            phase_ = phase
            shake = 1 + random.uniform(-self.shake_factor,self.shake_factor)
            flex = 1 + random.uniform(-self.flex_factor,self.flex_factor)
            for i in range(self.length):
                # initialize the value
                value = 0
                num = int(period* flex)
                seq_resampled = signal.resample(seq_, num)
                # calculate the phase of the square wave
                phase_ = (phase_ + 1) % num
                # determine the value of the wave at this phase
                value += seq_resampled[phase_] # high state
                # append it to the sequence
                if phase_ == 0:
                    shake = 1 + random.uniform(-self.shake_factor,self.shake_factor)
                    flex = 1 + random.uniform(-self.flex_factor,self.flex_factor)
                seq.append(value * shake)


        elif self.type == 'randseasonq':
            # generate a quantitive seasonal sequence
            if len(self.seq) > 0:
                # sequence is set by hand
                seq_ = self.seq
                period = self.period
            else:
                # sequence is generated automaticaly
                period = random.randint(int(self.period/3), self.period)
                t = TrendSequenceGenerator(amplitude=self.amplitude, delta_s=self.delta_s, length=period,type=self.limit_type)
                seq_ = t.generate()
            # generate
            
            phase = random.randint(0, period)
            phase_ = phase
            shake = 1 + random.uniform(-self.shake_factor,self.shake_factor)
            flex = 1 + random.uniform(-self.flex_factor,self.flex_factor)
            for i in range(self.length):
                # initialize the value
                value = 0
                num = int(period* flex)
                seq_resampled = signal.resample(seq_, num)
                # calculate the phase of the square wave
                phase_ = (phase_ + 1) % num
                # determine the value of the wave at this phase
                value += seq_resampled[phase_] # high state
                # append it to the sequence
                if phase_ == 0:
                    shake = 1 + random.uniform(-self.shake_factor,self.shake_factor)
                    flex = 1 + random.uniform(-self.flex_factor,self.flex_factor)
                seq.append(value * shake)
            seq = quantize(seq,self.levels)       

        else:
            # invalid type
            raise ValueError("Invalid type.")

        return seq