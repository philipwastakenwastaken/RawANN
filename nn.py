import numpy as np


# ACTIVATION FUNCTIONS #
def reLU(a):
    return np.maximum(a, 0)


# derirative
def reLU_prime(a):
    return np.greater(a, 0)


def softmax(a):
    exp = np.exp(a)
    return exp / np.sum(exp)


def quadratic_cost_deriv(a, y):
    return a - y


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

    def forward_tuple(self, a_n):
        z = np.dot(self.weights, a_n) + self.biases
        return (z, self.activator(z))


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
        # use reLU for now, maybe softmax later
        self.output_layer = Layer(output_layer_size,
                                  neurons_per_layer[-1],
                                  reLU)
        self.hidden_layers.append(self.output_layer)

    def forward_pass(self, X):
        self.input_layer = X

        a_n = self.input_layer
        for layer in self.hidden_layers:
            a_n = layer.forward(a_n)

        return a_n

    def mse_cost(self, output, expected):
        sub = np.subtract(expected, output)
        return np.linalg.norm(sub) ** 2

    def backprop(self, x, y):
        nabla_w = [np.zeros(layer.weights.shape)
                   for layer in self.hidden_layers]
        nabla_b = [np.zeros(layer.biases.shape)
                   for layer in self.hidden_layers]

        # save all activations from forward pass

        # input layer
        a_n = x
        activations = [a_n]
        z = []

        # hidden layers
        for layer in self.hidden_layers:
            zs, a_n = layer.forward_tuple(a_n)
            activations.append(a_n)
            z.append(zs)

        delta = quadratic_cost_deriv(activations[-1], y) * reLU_prime(z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        layers = self.hidden_layers
        for l in range(2, len(layers) + 1):  # +1 for input layer
            zs = z[-l]
            rp = reLU_prime(zs)
            delta = np.dot(layers[-l + 1].weights.T, delta) * rp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_w, nabla_b

    def update_mini_batches(self, mini_batch, learning_rate):
        # gradient for weight and biases
        nabla_w = [np.zeros(layer.weights.shape)
                   for layer in self.hidden_layers]

        nabla_b = [np.zeros(layer.biases.shape)
                   for layer in self.hidden_layers]

        # STOCHASTIC GRADIENT DESCENT
        for x, y in mini_batch:
            # delta for gradient; how much does individual data point
            # affect parameters?
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)

            # summing all the deltas here
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnw for nb, dnw in zip(nabla_b, delta_nabla_b)]

        # we now have an approximation for the gradient descent
        # from our batches.
        # we will now step in the opposite direction according to the
        # learning rate in order to find a local (hopefully global) minimum
        n = len(mini_batch)
        for l, (layer, nw, nb) in enumerate(zip(self.hidden_layers, nabla_w, nabla_b)):
            self.hidden_layers[l].weights = layer.weights - learning_rate / n * nw
            self.hidden_layers[l].biases = layer.biases - learning_rate / n * nb

    def fit(self, training_data, learning_rate, epochs=1, batch_size=16,
            validation_data=None):
        cols = len(training_data)

        for i in range(epochs):

            print("epoch:", i)
            # shuffle data every epoch
            np.random.shuffle(training_data)

            # split data into mini batches
            mini_batches = [training_data[i:cols+batch_size]
                            for i in range(0, cols, batch_size)]

            # update weights and biases for each batch
            for mini_batch in mini_batches:
                self.update_mini_batches(mini_batch, learning_rate)

            if validation_data:
                correct = 0
                for img, label in validation_data:
                    expected = self.forward_pass(img)
                    expected = np.argmax(expected)
                    gt = np.argmax(label)

                    if (gt == expected):
                        correct += 1
                print("correct:", correct, "/", len(validation_data))
