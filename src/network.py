import numpy as np

class Network:

    def __init__(self, sizes, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.weights = [
            rng.standard_normal(x, y) * np.sqrt(2.0 / x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

        self.biases = [np.zeros(1, y) for y in sizes[1:]]
