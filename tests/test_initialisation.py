"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .asserts import *
from ..ImportBWE import *

"""___Function______________________________________________________________"""

def test_layer():
    layer = Layer()
    layer.dense(2, 3)
    assertEqual(layer.weights.size, 6)
