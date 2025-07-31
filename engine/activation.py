"""___Modules_______________________________________________________________"""

# Python
import numpy as np

"""___Classes_______________________________________________________________"""

class Activation():      # Build for one inputs set (single) or multiple inputs set (multiple)

    def __init__(self) -> None:
        self.forwardict = {
            "relu": self.ReLU,
            "softmax": self.Softmax,
            "clip": self.Clip,
            "spread": self.Spread,
            "identity": self.Identity,
            "argmax": self.Argmax,
            "sigmoid": self.Sigmoid,
            "tansigmoid": self.TanSigmoid,

            "baffwill_v1": self.BaffWill_v1,
        }

    def forward(self,
                inputs: list[float],
                fct: str,
                parameters: list = []
                ) -> None:
        if type(fct) == str:
            self.forwardict[fct](inputs)
        else:
            self.output = fct(inputs, parameters)

    def ReLU(self, input: float) -> None:    # single
        self.output = np.maximum(0, input)

    def Softmax(self, inputs: np.array) -> None:     # multiple
        if len(inputs.shape) > 1:
            values = inputs - np.max(inputs, axis=1, keepdims=1)
            exp_values = np.exp(values)
            norm_base = np.sum(exp_values, axis=1, keepdims=1)
            self.output = exp_values / norm_base
        else:
            values = inputs - np.max(inputs)
            exp_values = np.exp(values)
            norm_base = np.sum(exp_values)
            self.output = exp_values / norm_base

    def Clip(self,
             input: np.array,
             mini: float,
             maxi: float
             ) -> None:
        # Keep values in I, the ones outside are brought back to maxi or mini
        self.output = np.clip(input, mini, maxi)

    def Spread(self,
               inputs: np.array,
               mini: float,
               maxi: float
               ) -> None:  # multiple
        # Dilatation of the output to interval mini maxi
        inputs_norm = (inputs - np.min(inputs)) * (maxi - mini)
        self.output = inputs_norm + mini

    def Identity(self, inputs: list[float]) -> None:
        self.output = inputs

    def Argmax(self, inputs: np.array) -> None:      # single
        if len(inputs.shape) == 1:
            self.output = np.argmax(inputs)
        else:
            self.output = np.argmax(inputs, axis=1)

    def Sigmoid(self,
                inputs: np.array,
                temperature: float = 1
                ) -> None:
        values = np.minimum(-inputs * temperature, 1e2)
        self.output = 1 / (1 + np.exp(values))

    def TanSigmoid(self,
                   inputs: np.array,
                   temperature: float = 1
                   ) -> None:     # single
        values = np.minimum(-2 * inputs * temperature, 1e2)
        self.output = 2 / (1 + np.exp(values)) - 1

    def BaffWill_v1(self,
                    inputs: list[float]
                    ) -> None:  # multiple
        # Sort input values from the highest to the lowest. From 0 to len(inputs)-1.
        # The ouput is the rankings of positions according to their inputs.
        lenI = len(inputs)
        ranking = [0] * lenI
        ranking[-1] = int(np.argmin(inputs))

        for value in range(lenI - 1):
            # print('Ranking : ' + str(ranking))
            # print('Inputs : ' + str(inputs))
            # print("")
            ranking[value] = int(np.argmax(inputs))
            inputs[ranking[value]] = inputs[ranking[-1]] - 1

        self.output = ranking
