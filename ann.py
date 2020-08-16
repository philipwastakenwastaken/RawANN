import sys
import numpy as np
import data_stream as ds

np.set_printoptions(threshold=sys.maxsize)
np.random.seed(12445512)

epoch_size = 1024

print("Reading data")

# random sample input data
X = ds.read_train_images(epoch_size)

y = ds.read_train_labels(epoch_size)


def reLU(a):
    return np.maximum(a, 0)


def softmax(a):
    exp = np.exp(a)
    return exp / np.sum(exp)


class Layer:

    def __init__(self, num_neurons, input_size, activator=reLU):
        # vector of biases
        self.biases = np.random.randn(num_neurons, 1)

        # (num_neurons x input_size) weights matrix
        self.weights = np.random.randn(num_neurons, input_size)

        self.activator = activator

    def forward(self, a_n):
        Wa = np.dot(self.weights, a_n) + self.biases
        return self.activator(Wa)


class Model:

    def __init__(self, neurons_per_layer, input_layer_size, output_layer_size):
        # input layer has no weights or biases, just vector of input
        self.input_layer = np.zeros(input_layer_size)

        # dense layers with weights and biases
        self.hidden_layers = []

        # construct layers
        self.hidden_layers.append(Layer(neurons_per_layer[0],
                                  input_layer_size))

        for index, neurons in enumerate(neurons_per_layer[1:]):
            # input size is number of neurons from previous layer
            # since list is sliced index - 1 is not needed
            layer = Layer(neurons, neurons_per_layer[index])
            self.hidden_layers.append(layer)

        # output layer also has weights and biases
        # uses softmax instead of reLU
        self.output_layer = Layer(output_layer_size,
                                  neurons_per_layer[-1],
                                  softmax)

    def forward_pass(self, X):
        self.input_layer = X

        a_n = self.input_layer
        for layer in self.hidden_layers:
            a_n = layer.forward(a_n)

        return self.output_layer.forward(a_n)

    def mse_cost(self, output, expected):
        sub = np.subtract(expected, output)
        return np.linalg.norm(sub) ** 2

    def fit(self, training_data, epochs=1, batch_size=1):
        cols = len(training_data)

        for _ in range(epochs):

            # shuffle data every epoch
            np.random.shuffle(training_data)

            for batch_start in range(0, cols, batch_size):

                cost = 0.0
                for col in range(batch_start, batch_start + batch_size):
                    (X_t, X_e) = training_data[col]
                    a_n = self.forward_pass(X_t)
                    cost += self.mse_cost(a_n, X_e)

                cost /= 2 * batch_size
                print(cost)

    def back_propogate(self):
        pass


print("Processing data")
p = []
for col in range(epoch_size):
    entry = (X[:, col].reshape(28 * 28, 1), y[:, col])
    p.append(entry)

model = Model([16, 16], 28 * 28, 10)

print("Fitting model")
model.fit(p, 1, 32)
