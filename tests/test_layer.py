"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .asserts import *
from ..ImportBWE import *

# Python
import pytest

"""___Function______________________________________________________________"""

def test_layer_init() -> None:
    layer = Layer()
    layer.dense(2, 3)
    assertEqual(layer.weights.size, 6)

def test_layer_print() -> None:
    layer = Layer()
    layer.dense(5, 4)
    layer.info()
    assertIsInstance(repr(layer), str)

def test_layer_copy() -> None:
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

def test_layer_straight() -> None:
    layer = Layer()
    layer.straight(2)
    assertEqual(layer.weights.size, 2)
    assertEqual(layer.biases.size, 2)
    assertEqual(layer.activation, "relu")
    assertEqual(layer.parameters, [])

def test_layer_random() -> None:
    layer = Layer()
    layer.random(5, 4)
    assertEqual(layer.weights.size, 20)
    assertEqual(layer.biases.size, 4)
    assertEqual(layer.activation, "relu")
    assertEqual(layer.parameters, [])

def test_layer_forward() -> None:
    layer = Layer()
    layer.dense(1, 1, init="test")
    layer.forward([4])
    assertListEqual(layer.output, [5])
    layer.dense(3, 4, init="test")
    layer.forward([1, 2, 3])
    assertListEqual(layer.output, [7, 7, 7, 7])
    layer.random(3, 4, init="test")
    layer.forward([1, 2, 3])
    assertListIsInstance(layer.output, float)
    layer.straight(3, init="test")
    layer.forward([1, 2, 3])
    assertListEqual(layer.output, [2, 3, 4])

def test_layer_error_init() -> None:
    layer = Layer()
    with pytest.raises(InitUnknownError):
        layer.dense(5, 6, init="vache")
    with pytest.raises(InitUnknownError):
        layer.straight(5, init="boeuf")
    with pytest.raises(InitUnknownError):
        layer.random(2, 2, init="mouton")

def test_layer_copy_from_network():
    net = Network()
    layer1 = Layer()
    net.default([2, 3, 2], init="test")
    layer1.copy_from_network(net, 1)
    layer2 = Layer()
    layer2.dense(3, 2, init="test")
    assertEqual(layer1, layer2)
