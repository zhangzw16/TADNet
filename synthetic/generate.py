from trend import *
from noise import * 
from season import *
import numpy as np


# to generate some full sequence
class generator:
    """
    A class to generate a full sequence with given infomation.

    args:
        info_pack: list, include all of the infomation we need.
    """
    def __init__(self, info_pack):
        self.info_pack = info_pack
        amplitude = self.info_pack[0]
        period = self.info_pack[1]
        length = self.info_pack[2]
        type = self.info_pack[3]
        wavnum = self.info_pack[4]
        seq = self.info_pack[5]
        delta_s = self.info_pack[6]
        shake_factor = self.info_pack[7]
        limit_type = self.info_pack[8]
        levels = self.info_pack[9]
        flex_factor = self.info_pack[10]
        t_amplitude = self.info_pack[11]
        t_delta_s = self.info_pack[12]
        n_amplitude = self.info_pack[13]
        self.t = TrendSequenceGenerator(amplitude=t_amplitude, delta_s=t_delta_s, length=length)
        self.n = NoiseSequenceGenerator(amplitude=n_amplitude, length=length, type='gauss')
        self.s = PeriodicSequenceGenerator(amplitude=amplitude, length=length, type=type, period=period, delta_s=delta_s, shake_factor=shake_factor, wavnum=wavnum, limit_type=limit_type, levels=levels, flex_factor=flex_factor)

    def generate(self):
        t_seq = np.array(self.t.generate())
        n_seq = np.array(self.n.generate())
        s_seq = np.array(self.s.generate())
        seq = t_seq + n_seq + s_seq
        return [t_seq, n_seq, s_seq, seq]


        
    