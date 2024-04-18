import random
import numpy as np
from trend import TrendSequenceGenerator
class AnomalySequenceGenerator:
    """
    A class to generate a sequence with anomaly subsignals

    args:
        amplitude: float, the amplitude parameter of the sequence. Default is 1.
        type: str, from 'point', 'interval', 'contextual', 'collective', 'shapelet', 'noise' 
    """

    def __init__(self, amplitude=1, type=('point',1)):
        self.amplitude = amplitude
        self.type = type

    def generate(self, time_series):
        # point anomaly
        if self.type[0] == 'point':
            point_num = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude * 10
            for i in range(point_num):
                index = random.randint(0, length - 1)
                amp_ = random.uniform(amp/10, amp)
                time_series[index] *= amp_ + 1 
        
        # interval anomaly
        elif self.type[0] == 'interval':
            interval_len = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude * 10
            start = random.randint(0, length - 1 - interval_len)
            end = start + interval_len
            m = np.mean(time_series[start:end])
            amp_ = random.uniform(amp/10, amp)
            time_series[start: end] += (amp_ + 1) * m
        
        # contextual anomaly
        elif self.type[0] == 'contextual':
            interval_len = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude * 10
            start = random.randint(0, length - 1 - interval_len)
            end = start + interval_len
            m = np.mean(time_series[start:end])
            amp_ = random.uniform(amp/10, amp)
            time_series[start: end] *= (amp_ + 1)

        # collective anomaly
        elif self.type[0] == 'collective':
            interval_len = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude * 10
            start = random.randint(0, length - 1 - interval_len)
            end = start + interval_len
            delta_s = 10 ** random.uniform(-5,-2)
            if start == 0:
                delta = 0
            else:
                delta =  time_series[start] - time_series[start-1]
            type_ = random.randint(0,3)
            type_list = ['amp_limited','amp_unlimited','amp_unlimited','amp_unlimited']
            type_str = type_list[type_]
            t = TrendSequenceGenerator(type=type_str, length=interval_len,amplitude=time_series[start],delta_s=delta_s,delta = 0,const=1)
            t_series = t.generate()
            for index in range(start, end):
                time_series[index] = t_series[index-start]
        
        # shapelet anomaly 
        elif self.type[0] == 'shapelet':
            interval_len = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude
            amp_ = random.uniform(amp/10, amp)
            start = random.randint(0, length - 1 - interval_len)
            end = start + interval_len
            sigma = np.std(time_series[start:end])
            for index in range(start, end):
                time_series[index] += random.uniform(-amp_, amp_)

        # noise anomaly
        elif self.type[0] == 'noise':
            interval_len = self.type[1]
            length = time_series.shape[0]
            amp = self.amplitude * 10
            amp_ = random.uniform(amp/5, amp)
            start = random.randint(0, length - 1 - interval_len)
            end = start + interval_len
            sigma = np.std(time_series[start:end])
            for index in range(start, end):
                time_series[index] += random.gauss(0, amp_)
        return time_series
        





