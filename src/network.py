import numpy as np

class Network:

    def __init__(self, sizes, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.weights = [
            rng.standard_normal((y, x)) * np.sqrt(2.0 / x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

        self.biases = [np.zeros(y) for y in sizes[1:]]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        return output_activations-y

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid function
    s = sigmoid(z)
    return s * (1 - s)
