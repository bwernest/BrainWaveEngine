"""___Modules_______________________________________________________________"""

# Python
import copy
import numpy as np
from matplotlib import pyplot as plt

# BrainWaveEngine
from .activation import Activation
from .layer import Layer

"""___Classes_______________________________________________________________"""

class Network():

    def __init__(self) -> None:
        self.brain: list[dict] = []
        self.len = 0
        self.trained = False

    def copy(self, net: "Network") -> None:
        self.brain = copy.deepcopy(net.brain)
        self.len = net.len
        self.trained = net.trained
        self.layers_list = net.layers_list

    def default(self,
                layers_list: list[int],
                factors_list: list[float] = [],
                init: str = "default"
                ) -> None:
        """
        Generates a neural network with the layers indicated. All activation functions are ReLU except for the output layer that uses Sigmoid.

        Parameters
        ----------
        layers_list : list
            Indicates the number of layer and the number of neuron each one contains. \
                For example (2, 4, 3) indicates a input layer of 2 neurons, one hidden layer of 4 neurons and a output layer of 3 neurons.
        factors_list : list, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        factors_list = np.array(factors_list)
        factor_type = 0
        self.__init__()
        self.layers_list = [layers_list[0]]

        if len(factors_list.shape) == 1 and len(factors_list) != 0:
            factors_w = factors_list[0]
            factors_b = factors_list[1]
            factor_type = 1

        elif len(factors_list.shape) == 2:
            factors_w = factors_list[:, 0]
            factors_b = factors_list[:, 1]
            factor_type = 2

        layer = Layer()
        for n in range(len(layers_list) - 1):

            if factor_type == 2:
                layer.dense(layers_list[n], layers_list[n + 1], 'relu',
                            k_weights=factors_w[n], k_biases=factors_b[n], init=init)
            elif factor_type == 1:
                layer.dense(layers_list[n], layers_list[n + 1], 'relu',
                            k_weights=factors_w, k_biases=factors_b, init=init)
            else:
                layer.dense(layers_list[n], layers_list[n + 1], 'relu', init=init)
            self.add_layer(layer)
        self.brain[self.len - 1][3] = 'sigmoid'
        self.brain[self.len - 1][4] = [1]

    def function(self,
                 start: float,
                 end: float,
                 n_segment: int
                 ) -> None:
        self.len = 3
        self.layers_list = (1, n_segment, n_segment, 1)
        self.default(self.layers_list)
        layer = Layer()
        layer.straight(n_segment)
        self.modify_layer(layer, 1)

        X = np.linspace(start, end, n_segment + 1)
        step = (end - start) / n_segment
        w, b = self.get_weights(), self.get_biases()
        w[:, 0], b[:, 0] = np.zeros(3 * n_segment), np.zeros(2 * n_segment + 1)

        w[:n_segment, 0] = np.abs(np.random.randn(n_segment) / step)
        w[n_segment:2 * n_segment, 0] = -1
        w[2 * n_segment:3 * n_segment, 0] = -1 * np.sign(w[:n_segment, 0])

        b[:n_segment, 0] = -abs(X[:-1]) * w[:n_segment, 0] * np.sign(X[:-1])
        b[n_segment:2 * n_segment, 0] = w[:n_segment, 0] * step * np.sign(w[:n_segment, 0])

        self.insert_weights(w)
        self.insert_biases(b)

    def fct(self,
            start: float,
            end: float,
            n_segment: int
            ) -> None:
        self.len = 3
        self.layers_list = (1, n_segment, n_segment, 1)
        self.default(self.layers_list)
        layer = Layer()
        layer.straight(n_segment)
        self.modify_layer(layer, 1)

        X = np.linspace(start, end, n_segment + 1)
        a_list, b_list = X[:-1], X[1:]
        c_list, d_list = np.random.randn(n_segment), np.random.randn(n_segment)
        e_list = np.divide(np.subtract(d_list, c_list), np.subtract(b_list, a_list))

        self.X = X
        self.a = a_list
        self.b = b_list
        self.c = c_list
        self.d = d_list
        self.e = e_list

        w = self.get_weights()
        b = self.get_biases()

        w[:, 0] = np.zeros(3 * n_segment) + 9.11
        b[:, 0] = np.zeros(2 * n_segment + 1) + 9.11

        w[:n_segment, 0] = np.abs(e_list)
        w[n_segment:2 * n_segment, 0] = -1
        w[2 * n_segment:, 0] = (np.zeros(n_segment) - 1) * np.sign(e_list)

        b[:n_segment, 0] = -np.abs(np.multiply(a_list, e_list)) * np.sign(a_list)
        b[n_segment:2 * n_segment, 0] = np.multiply(d_list - c_list, np.sign(e_list))
        b[2 * n_segment:, 0] = np.zeros(n_segment) + np.sum(d_list)

        self.w = w
        self.b = b

        self.insert_weights(w)
        self.insert_biases(b)

        '''
        weights[0, 0] = abs(e1)
        biases[0, 0] = -abs(a1*e1) * np.sign(a1)

        weights[2, 0] = -1
        weights[4, 0] = -1 * np.sign(e1)

        biases[2, 0] = (d1-c1) * np.sign(e1)
        '''

    def add_layer(self,
                  layer: Layer
                  ) -> None:
        self.brain.append({})
        self.len += 1
        self.layers_list += (layer.neurons,)
        self.brain[self.len - 1]["type"] = layer.type
        self.brain[self.len - 1]["weights"] = layer.weights
        self.brain[self.len - 1]["biases"] = layer.biases
        self.brain[self.len - 1]["activation"] = layer.activation
        self.brain[self.len - 1]["parameters"] = layer.parameters

    def modify_layer(self,
                     layer: Layer,
                     n_layer: int
                     ) -> None:
        """Modifies layer n°n_layer, starting from 0."""
        self.brain[n_layer]["type"] = layer.type
        self.brain[n_layer]["weights"] = layer.weights
        self.brain[n_layer]["biases"] = layer.biases
        self.brain[n_layer]["activation"] = layer.activation
        self.brain[n_layer]["parameters"] = layer.parameters

    def info(self) -> None:
        for n_layer in range(self.len):
            print('\nLayer', n_layer, ':')
            layer = Layer()
            layer.type = self.brain[n_layer]["type"]
            layer.weights = self.brain[n_layer]["weights"]
            layer.biases = self.brain[n_layer]["biases"]
            layer.activation = self.brain[n_layer]["activation"]
            layer.parameters = self.brain[n_layer]["parameters"]
            layer.info()

    def forward(self,
                inputs: list[float]
                ) -> None:     # Activation function ReLU except the last layer
        values = inputs
        layer = Layer()
        for n_layer in range(self.len):
            layer.inputs = self.layers_list[n_layer]
            layer.neurons = self.layers_list[n_layer + 1]
            layer.type = self.brain[n_layer]["type"]
            layer.weights = self.brain[n_layer]["weights"]
            layer.biases = self.brain[n_layer]["biases"]
            layer.activation = self.brain[n_layer]["activation"]
            layer.parameters = self.brain[n_layer]["parameters"]
            layer.forward(values)
            values = layer.output

        self.output = values

    def get_activations(self) -> list[str]:
        activation_list = []
        for layer in range(self.len):
            activation_list.append(self.brain[layer]["activation"])
        return activation_list

    def get_weights(self) -> np.array:
        """
        Useful function that helps to inspect all weights of a network.
        Parameters
        ----------
        brain : list[dict]
            Brain from the network we want to inspect.

        Returns
        -------
        weights_list : list
            List with one line per weight and two columns. The first columns contains the weights' values.\
                The second column contains the coordinates of the corresponding weight in its brain.
        """
        weights_list = []
        for layer in range(self.len):
            for neuron in range(len(self.brain[layer]["weights"])):
                for weight in range(len(self.brain[layer]["weights"][neuron])):
                    weight_value = self.brain[layer]["weights"][neuron][weight]
                    weights_list.append([weight_value, layer, neuron, weight])
        return np.array(weights_list)

    def get_biases(self) -> np.array:
        """
        Useful function that helps to inspect all biases of a network.
        Parameters
        ----------
        brain : list
            Brain from the network we want to inspect.

        Returns
        -------
        weights_list : list
            List with one line per bias and two columns. The first columns contains the biases' values.\
                The second column contains the coordinates of the corresponding bias in its brain.
        """
        biases_list = []
        for layer in range(self.len):
            for bias in range(len(self.brain[layer]["biases"])):
                bias_value = self.brain[layer]["biases"][bias]
                biases_list.append([bias_value, layer, bias])
        return np.array(biases_list)

    def insert_activations(self,
                           activation_list: list[Activation]
                           ) -> None:
        for layer in range(self.len):
            self.brain[layer]["activation"] = activation_list[layer]

    def insert_weights(self,
                       weights_list: list[list[float]]
                       ) -> None:
        if type(weights_list) == list:
            weights_list = np.array(weights_list)
        for weight in range(len(weights_list)):
            x, y, z = np.array(weights_list[weight, 1:], dtype='uint8')
            self.brain[x]["weights"][y][z] = weights_list[weight, 0]

    def insert_biases(self,
                      biases_list: list[list[float]]
                      ) -> None:
        if type(biases_list) == list:
            biases_list = np.array(biases_list)
        for bias in range(len(biases_list)):
            x, y = np.array(biases_list[bias, 1:], dtype='uint8')
            self.brain[x]["biases"][y] = biases_list[bias, 0]

    def display(self,
                figure: int = 6
                ) -> None:
        """
        Display the neural network on the precised figure.

        Parameters
        ----------
        figure : int, optional
            Number of figure where the network is displayed. The default is 6.

        Returns
        -------
        None.
        """
        plt.close(fig=figure)
        plt.figure(figure)
        '''___Collect_data___'''
        self.layers_list = np.zeros(self.len + 1, dtype='int')
        self.layers_list[0] = len(self.brain[0]["weights"])
        for n_layer in range(self.len):
            self.layers_list[n_layer + 1] = len(self.brain[n_layer]["biases"])

        '''___Neurons___'''
        X = np.linspace(0, len(self.layers_list) - 1, len(self.layers_list))
        Y = []
        for n_neuron in range(len(self.layers_list)):
            y_neurons = np.linspace(0, self.layers_list[n_neuron] - 1, self.layers_list[n_neuron])
            Y.append(y_neurons - np.mean(y_neurons))

        '''___Synapses___'''
        for n_layer in range(len(self.layers_list) - 1):

            if self.brain[n_layer]["type"] == 'dense':
                for n_neuron in range(len(Y[n_layer])):
                    for n_synapse in range(len(Y[n_layer + 1])):
                        plt.plot([X[n_layer], X[n_layer + 1]], [Y[n_layer]
                                 [n_neuron], Y[n_layer + 1][n_synapse]], c='black')

            if self.brain[n_layer]["type"] == 'straight':
                for n_neuron in range(len(Y[n_layer])):
                    plt.plot([X[n_layer], X[n_layer + 1]], [Y[n_layer][n_neuron], Y[n_layer][n_neuron]], c='black')

            if self.brain[n_layer]["type"] == 'random':
                for n_neuron in range(len(Y[n_layer])):
                    for n_synapse in range(len(Y[n_layer + 1])):
                        if not np.isnan(self.brain[n_layer]["weights"][n_neuron][n_synapse]):
                            plt.plot([X[n_layer], X[n_layer + 1]], [Y[n_layer]
                                     [n_neuron], Y[n_layer + 1][n_synapse]], c='black')

        '''___Plot___'''
        abscissa, ordinate = [], []
        for n_layer in range(len(self.layers_list)):
            for n_neuron in range(self.layers_list[n_layer]):
                abscissa.append(X[n_layer])
                ordinate.append(Y[n_layer][n_neuron])

        plt.scatter(abscissa, ordinate, c=abscissa, cmap='brg', s=300)
        plt.show()

    def test(self,
             mini: float,
             maxi: float,
             tries: int = 10000
             ) -> None:
        samples = (np.random.random((tries, len(self.brain[0]["weights"]))) + mini) * (maxi - mini)
        self.forward(samples)
        plt.close(fig=10)
        plt.figure(10)
        outmin, outmax = np.min(self.output), np.max(self.output)
        outputs = len(self.brain[-1]["biases"])
        for output in range(outputs):
            colour = output / (outputs)
            plt.scatter(np.linspace(1, tries, tries) + tries * output,
                        self.output[:, output], s=5, color=(colour, 0, 0.5), label='Output n°' + str(output))
            plt.vlines(tries * output, outmin, outmax, ls='dashed', colors='black')
        plt.vlines(tries * outputs, outmin, outmax, ls='dashed', colors='black')
        plt.legend()
        plt.show()
