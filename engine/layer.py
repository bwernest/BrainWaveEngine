"""___Modules_______________________________________________________________"""

# Python
import numpy as np
import copy

# BrainWaveEngine
from .activation import Activation

"""___Layer_generation______________________________________________________"""

class Layer() :
    """
    Information
    ----------
    Class used to generate layers. This class can only creates one layer at a time. To create multiples layers, please use the Network class.
    """

    n_inputs : int
    n_neurons : int

    def __repr__(self) -> str:
        log = f"<Layer w/ {len(self.weights)} inputs {len(self.biases)} outputs>"
        
        return log
    
    def dense(self,
              n_inputs : int,
              n_neurons : int,
              activation : str = "relu",
              parameters : list = [],
              init : str = "default",
              k_weights : float = 1,
              k_biases : float = 1
              ) -> None :
        self.type = "dense"
        self.inputs = n_inputs
        self.neurons = n_neurons
        self.activation = activation
        self.parameters = parameters
        
        if init == "default" :
            self.weights = k_weights * np.random.randn(self.inputs, self.neurons)
            self.biases = k_biases * np.random.randn(self.neurons)
        
        elif init == "Xavier" :
            self.weights = 2
    
    def straight(self,
                 n_inputs : int,
                 activation : str = "relu",
                 parameters : list = [],
                 init : str = "default",
                 k_weights : float = 1,
                 k_biases : float = 1
                 ) -> None :
        self.type = 'straight'
        self.inputs = n_inputs
        self.neurons = n_inputs
        self.activation = activation
        self.parameters = parameters
        
        if init == "default" :
            self.weights = k_weights * np.array([np.random.randn(self.neurons)])
            self.biases = k_biases * np.random.randn(self.neurons)
    
    def random(self,
               n_inputs : int,
               n_neurons : int,
               activation : str = "relu",
               parameters : list = [],
               init : str = "default",
               k_weights : float = 1,
               k_biases : float = 1
               ) -> None :
        self.type = "random"
        self.inputs = n_inputs
        self.neurons = n_neurons
        self.activation = activation
        self.parameters = parameters
        ok = 0
        
        if init == 'default' :
            while ok != n_inputs + n_neurons :
                ok = 0
                self.weights = k_weights * np.random.randn(n_inputs, n_neurons)
                sort = self.weights * np.random.randint(0, 2, (n_inputs, n_neurons))
                for inputs in range(n_inputs) :
                    nones = np.mean(sort[inputs]==0)
                    if not nones in [0, 1] :
                        ok += 1
                if ok == n_inputs :
                    for outputs in range(n_neurons) :
                        nones = np.mean(sort[:, outputs]==0)
                        if not nones in [0, 1] :
                            ok += 1
            sort[sort==0] = None
            self.weights = sort
            self.biases = k_biases * np.random.randn(n_neurons)
    
    def forward(self,
                inputs : list[float]
                ) -> None :
        if self.inputs == 1 :
            self.output = np.zeros((len(inputs), self.neurons))
            for sample in range(len(inputs)) :
                self.output[sample] = inputs[sample] * self.weights + self.biases
        elif self.type == 'dense' :
            self.output = np.dot(inputs, self.weights) + self.biases
        elif self.type == 'straight' :
            self.output = inputs * self.weights + self.biases
        elif self.type == 'random' :
            weights = copy.deepcopy(self.weights)
            weights[np.isnan(weights)] = 0
            self.output = np.dot(inputs, weights) + self.biases
        activation = Activation()
        activation.forward(self.output, self.activation, parameters=self.parameters)
        self.output = activation.output
    
    def copy(self,
             network : object,
             n_layer : int
             ) -> None :
        self.inputs = network.layers_list[n_layer]
        self.neurons = network.layers_list[n_layer+1]
        self.type = network.brain[n_layer][0]
        self.weights = network.brain[n_layer][1]
        self.biases = network.brain[n_layer][2]
        self.activation = network.brain[n_layer][3]
        self.parameters = network.brain[n_layer][4]
    
    def info(self) -> None :
        print(f"Type : {self.type}\nActivation : {self.activation}\n"+
              f"Parameters : {self.parameters}\nWeights : {self.weights}\n"+
              f"Biases : {self.biases})")
