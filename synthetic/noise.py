import random

class NoiseSequenceGenerator:
    """
    A class to generate a noise sequence with a given amplitude.

    args:
        amplitude: float, the amplitude parameter of the noise sequence. Default is 1.
        length: int, the length of the noise sequence. Default is 10.
    """

    def __init__(self, amplitude=1, length=10, n_rate=1, type='uniform'):
        self.amplitude = amplitude
        self.length = length
        self.type = type
        self.n_rate = n_rate

    def generate(self):
        # generate a sequence of random values in [-amplitude, amplitude]
        n_len = int(self.length * self.n_rate)
        start = random.randint(0, self.length - n_len)
        end = start + n_len - 1
        if self.type == 'uniform':
            # uniform noise
            seq = [random.uniform(-self.amplitude, self.amplitude)/100 for _ in range(self.length)]
            for i in range(start, end):
                seq[i] = random.uniform(-self.amplitude, self.amplitude)


        elif self.type == 'gauss':
            # guass noise
            seq = [random.gauss(0, self.amplitude)/100 for _ in range(self.length)]
            for i in range(start, end):
                seq[i] = random.gauss(0, self.amplitude)
            

        else:
            # invalid type
            raise ValueError("Invalid type. Choose from 'uniform' or 'gauss'.")
        
    # to do: partial noise


        return seq