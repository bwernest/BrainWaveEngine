"""___Modules_______________________________________________________________"""

# Python
from copy import deepcopy
import numpy as np

# BrainWaveEngine
from .activation import Activation
from ..tools.errors import *

"""___Classes_______________________________________________________________"""

class Layer(Activation):
    """
    Information
    ----------
    Class used to generate layers. This class can only creates one layer at a time. To create multiples layers, please use the Network class.

    Attributs
    ----------
    n_inputs : int
        Nombre de neurones en amont de cette couche.
    n_neurons : int
        Nombre de neurones de cette couche.
    activation : str
        Fonction d'activation des neurones.
    weights : np.ndarray
        Tableau des weights.
    biases : np.ndarray
        Tableau des biases.
    parameters : list
        Liste de paramètres utiles pour certaines fonctions d'activation ou autres.
    init : str
        Méthode d'initialisation des weights et biases.
    """

    n_inputs: int
    n_neurons: int

    def __init__(self) -> None:
        super().__init__()
        self.forwardict = {
            "dense": self.forward_dense,
            "straight": self.forward_straight,
            "random": self.forward_random,
        }

    def __repr__(self) -> str:
        log = f"<Layer w/ {len(self.weights)} inputs {len(self.biases)} outputs>"
        return log

    def __eq__(self, layer: "Layer") -> bool:
        valids = [
            self.activation == layer.activation,
            self.n_inputs == layer.n_inputs,
            self.n_neurons == layer.n_neurons,
            self.parameters == self.parameters,
            self.type == layer.type,
            np.array_equal(self.biases, layer.biases),
            np.array_equal(self.weights, layer.weights),
        ]
        return np.mean(valids) == 1

    def __ne__(self, layer: "Layer"):
        return not self == layer

    def dense(self,
              n_inputs: int,
              n_neurons: int,
              activation: str = "relu",
              parameters: list = [],
              init: str = "default",
              k_weights: float = 1,
              k_biases: float = 1
              ) -> None:
        self.type = "dense"
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.parameters = parameters

        # Different methods of initialisation
        if init == "default":
            self.weights = k_weights * np.random.randn(n_inputs, n_neurons)
            self.biases = k_biases * np.random.randn(n_neurons)

        elif init == "test":
            self.weights = np.ones(shape=(n_inputs, n_neurons))
            self.biases = np.ones(shape=(n_neurons))

        else:
            raise InitUnknownError()

    def straight(self,
                 n_inputs: int,
                 activation: str = "relu",
                 parameters: list = [],
                 init: str = "default",
                 k_weights: float = 1,
                 k_biases: float = 1
                 ) -> None:
        self.type = 'straight'
        self.n_inputs = n_inputs
        self.n_neurons = n_inputs
        self.activation = activation
        self.parameters = parameters

        if init == "default":
            self.weights = k_weights * np.array([np.random.randn(n_inputs)])
            self.biases = k_biases * np.random.randn(n_inputs)

        elif init == "test":
            self.weights = np.ones(shape=(n_inputs))
            self.biases = np.ones(shape=(n_inputs))

        else:
            raise InitUnknownError()

    def random(self,
               n_inputs: int,
               n_neurons: int,
               activation: str = "relu",
               parameters: list = [],
               init: str = "default",
               k_weights: float = 1,
               k_biases: float = 1
               ) -> None:
        self.type = "random"
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.parameters = parameters
        ok = 0

        while ok != n_inputs + n_neurons:
            ok = 0
            self.weights = k_weights * np.random.randn(n_inputs, n_neurons)
            sort = self.weights * np.random.randint(0, 2, (n_inputs, n_neurons))
            for inputs in range(n_inputs):
                nones = np.mean(sort[inputs] == 0)
                if not nones in [0, 1]:
                    ok += 1
            if ok == n_inputs:
                for outputs in range(n_neurons):
                    nones = np.mean(sort[:, outputs] == 0)
                    if not nones in [0, 1]:
                        ok += 1
        sort[sort == 0] = None
        self.weights = sort
        self.biases = k_biases * np.random.randn(n_neurons)

        if init == 'default':
            for line in range(len(self.weights)):
                for w in range(len(self.weights[0])):
                    if self.weights[line, w] != None:
                        self.weights[line, w] = k_weights * np.random.random()
            self.biases = k_biases * np.random.randn(n_neurons)

        elif init == "test":
            for line in range(len(self.weights)):
                for w in range(len(self.weights[0])):
                    if self.weights[line, w] != None:
                        self.weights[line, w] = 1.
            self.biases = k_biases * np.ones(shape=(n_neurons))

        else:
            raise InitUnknownError()

    def forward(self,
                inputs: list[float]
                ) -> None:
        assert len(inputs) == self.n_inputs, WrongInputSize(f"Vecteur de taille {inputs} != {self.n_inputs}.")
        self.activate(self.forwardict[self.type](inputs))

    def forward_dense(self, inputs: np.ndarray) -> np.ndarray:
        return np.dot(inputs, self.weights) + self.biases

    def forward_straight(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * self.weights + self.biases

    def forward_random(self, inputs: np.ndarray) -> np.ndarray:
        weights = deepcopy(self.weights)
        weights[np.isnan(weights)] = 0
        return np.dot(inputs, weights) + self.biases

    def activate(self, inputs: np.ndarray) -> None:
        self.output = self.ActivDict[self.activation](inputs, self.parameters)

    def copy_from_layer(self, layer: "Layer") -> None:
        for key, elem in layer.__dict__.items():
            self.__dict__[key] = deepcopy(elem)

    def copy_from_network(
        self,
        network: object,
        n_layer: int
    ) -> None:
        self.copy_from_layer(network.brain[n_layer])

    def info(self) -> None:
        print(f"Type : {self.type}\nActivation : {self.activation}\n" +
              f"Parameters : {self.parameters}\nWeights : {self.weights}\n" +
              f"Biases : {self.biases})")
