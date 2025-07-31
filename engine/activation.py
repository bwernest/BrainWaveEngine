"""___Modules_______________________________________________________________"""

# Python
import numpy as np

"""___Classes_______________________________________________________________"""

class Activation():      # Build for one inputs set (single) or multiple inputs set (multiple)

    def __init__(self) -> None:
        self.ActivDict: dict[str, function] = {
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

    def ReLU(self, input: float, *args) -> float:    # single
        return np.maximum(0, input)

    def Softmax(self, inputs: np.ndarray, *args) -> np.ndarray:     # multiple
        if len(inputs.shape) > 1:
            values = inputs - np.max(inputs, axis=1, keepdims=1)
            exp_values = np.exp(values)
            norm_base = np.sum(exp_values, axis=1, keepdims=1)
            return exp_values / norm_base
        else:
            values = inputs - np.max(inputs)
            exp_values = np.exp(values)
            norm_base = np.sum(exp_values)
            return exp_values / norm_base

    def Clip(self, input: np.ndarray, *args) -> np.ndarray:
        # Keep values in I, the ones outside are brought back to maxi or mini
        return np.clip(input, args[0][0], args[0][1])

    def Spread(self, inputs: np.ndarray, *args) -> np.ndarray:  # multiple
        # Dilatation of the output to interval mini maxi
        return (inputs - np.min(inputs)) * (args[0][1] - args[0][0]) + args[0][0]

    def Identity(self, inputs: np.ndarray, *args) -> np.ndarray:
        return inputs

    def Argmax(self, inputs: np.ndarray, *args) -> int:      # single
        if len(inputs.shape) == 1:
            return np.argmax(inputs)
        else:
            return np.argmax(inputs, axis=1)

    def Sigmoid(self, inputs: np.array, *args) -> np.ndarray:
        values = np.minimum(-inputs * args[0][0], 1e2)
        return 1 / (1 + np.exp(values))

    def TanSigmoid(self, inputs: np.array, *args) -> np.ndarray:     # multiple
        values = np.minimum(-2 * inputs * args[0][0], 1e2)
        return 2 / (1 + np.exp(values)) - 1

    def BaffWill_v1(self, inputs: list[float], *args) -> np.ndarray:  # multiple
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

        return ranking
