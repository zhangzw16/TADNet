import random

class TrendSequenceGenerator:
    """
    A class to generate a sequence with second-order difference absolute value not larger than x.

    args:
        amplitude: float, the amplitude parameter of the sequence. Default is 1.
        length: int, the length of the sequence. Default is 10.
        delta_s: float, the second difference parameter of the the sequence. Default is 1.
    """

    def __init__(self, amplitude=1, delta_s=1, length=100, type='amp_unlimited'):
        self.amplitude = amplitude
        self.delta_s = delta_s 
        self.length = length
        self.type = type

    def generate(self):
        if self.type == 'amp_unlimited':
            # initialize the sequence with two random values in [-delta_s, delta_s]
            seq = [random.uniform(-self.amplitude, self.amplitude) for _ in range(1)]
            delta = 0

            # generate the rest of the sequence by adding a random value in [-x, x] to the previous two values
            for i in range(1, self.length):
                delta = delta + random.uniform(-self.delta_s, self.delta_s) # a random value in [-x, x]
                seq.append(seq[i-1] + delta) # add it to the previous value

        elif self.type == 'amp_limited':
            done = 0
            count = 0
            while not done:
                count += 1
                # initialize the sequence with two random values in [-delta_s, delta_s]
                seq = [random.uniform(-self.amplitude, self.amplitude) for _ in range(1)]
                delta = 0
                # generate the rest of the sequence by adding a random value in [-x, x] to the previous two values
                for i in range(1, self.length):
                    delta = delta + random.uniform(-self.delta_s, self.delta_s) # a random value in [-x, x]
                    seq.append(seq[i-1] + delta) # add it to the previous value
                if abs(seq[-1] - seq[0]) <= self.delta_s*self.length / 50:
                    done = 1
        
        else:
            # invalid type
            raise ValueError("Invalid type.")

        return seq