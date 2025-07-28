"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .asserts import *
from ..ImportBWE import *

"""___Function______________________________________________________________"""

def test_layer_init():
    layer = Layer()
    layer.dense(2, 3)
    assertEqual(layer.weights.size, 6)

def test_layer_print():
    layer = Layer()
    layer.dense(5, 4)
    layer.info()
    assertIsInstance(repr(layer), str)

def test_layer_copy():
    layer1 = Layer()
    layer1.dense(2, 6)
    layer2 = Layer()
    layer2.copy_from_layer(layer1)
    layer3 = Layer()
    layer3.copy_from_layer(layer2)
    layer3.weights = "Rideau"
    layer4 = Layer()
    layer4.dense(3, 6)
    print("Original")
    print(layer1.weights)
    layer5 = Layer()
    layer5.copy_from_layer(layer1)
    layer5.weights[0][0] += 1
    print("New")
    print(layer1.weights)
    print(layer5.weights)
    assertEqual(layer1, layer2)
    assertNotEqual(layer2, layer3)
    assertNotEqual(layer1, layer4)
    assertNotEqual(layer1, layer5)
