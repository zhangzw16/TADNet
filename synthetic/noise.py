import random

class NoiseSequenceGenerator:
    """
    A class to generate a noise sequence with a given amplitude.

    args:
        amplitude: float, the amplitude parameter of the noise sequence. Default is 1.
        length: int, the length of the noise sequence. Default is 10.
    """

    def __init__(self, amplitude=1, length=10, type='uniform'):
        self.amplitude = amplitude
        self.length = length
        self.type = type

    def generate(self):
        # generate a sequence of random values in [-amplitude, amplitude]
        if self.type == 'uniform':
            # uniform noise
            seq = [random.uniform(-self.amplitude, self.amplitude) for _ in range(self.length)]

        if self.type == 'gauss':
            # guass noise
            seq = [random.gauss(0, self.amplitude) for _ in range(self.length)]

        else:
            # invalid type
            raise ValueError("Invalid type. Choose from 'uniform' or 'gauss'.")


        return seq