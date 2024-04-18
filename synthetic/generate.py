from trend import *
from noise import * 
from season import *
from anomaly import *
from const import a_config
import numpy as np
import random

# to generate some full sequence
class generator:
    """
    A class to generate a full sequence with given infomation.

    args:
        info_pack: list, include all of the infomation we need.
    """
    def __init__(self, info_pack=None, random_ = False):
        if info_pack != None:
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
            noise_type = self.info_pack[9]
            levels = self.info_pack[10]
            flex_factor = self.info_pack[11]
            t_amplitude = self.info_pack[12]
            t_delta_s = self.info_pack[13]
            n_amplitude = self.info_pack[14]
            n_rate = self.info_pack[15]
            

        if random_:
            amplitude = random.uniform(0.5,20)
            period = int(2 * 10 ** random.uniform(1.5,3))
            length = 4000 * random.randint(2,3)
            type_ = random.randint(0,4)
            type_list = ['randsine','randseason','randseasonq','randseason','randseason']
            type = type_list[type_]
            wavnum = random.randint(1,8)
            delta_s = 10 ** random.uniform(-5,-2)
            shake_factor = 10 ** random.uniform(-7,-4)
            
            type_ = random.randint(0,2)
            type_list = ['amp_limited','amp_limited','amp_unlimited']
            limit_type = type_list[type_]

            type_ = random.randint(0,1)
            type_list = ['gauss','uniform']
            noise_type = type_list[type_]
            levels = random.randint(10,200)
            flex_factor = 2 * 10 ** random.uniform(-6,-4)
            t_amplitude = random.uniform(0.1,100)
            t_delta_s = 10 ** random.uniform(-7,-3)
            n_amplitude = 10 ** random.uniform(-7,-1)
            n_rate = 1


        self.t = TrendSequenceGenerator(amplitude=t_amplitude, delta_s=t_delta_s, length=length)
        self.n = NoiseSequenceGenerator(amplitude=n_amplitude, length=length, type=noise_type, n_rate=n_rate)
        self.s = PeriodicSequenceGenerator(amplitude=amplitude, length=length, type=type, period=period, delta_s=delta_s, shake_factor=shake_factor, wavnum=wavnum, limit_type=limit_type, levels=levels, flex_factor=flex_factor)
    
    def add_anomaly(self, series, config):
        size = len(config)
        num = 0
        seed = random.randint(0,2000)
        for i in range(size):
            num_b = num 
            num = config[i][2]
            if seed >= num_b and seed < num:
                amp= config[i][0]
                a_type = config[i][1]
                ano = AnomalySequenceGenerator(amp, a_type)
                res = ano.generate(series)
                return res
        return series




    def generate(self):
        t_seq = np.array(self.t.generate())
        n_seq = np.array(self.n.generate())
        s_seq = np.array(self.s.generate())
        s_seq = s_seq - np.mean(s_seq)
        seq = self.add_anomaly(t_seq+ s_seq, a_config) + n_seq 
        return [t_seq, n_seq, s_seq, seq]


        
    