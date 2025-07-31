"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .asserts import *
from ..ImportBWE import *

"""___Function______________________________________________________________"""

def test_NetInit():
    net = Network()
    assertEqual(net.brain, [])

def test_NetEqual() -> None:
    net1 = Network()
    net1.default((4, 5, 5, 4))
    net2 = Network()
    net2.copy(net1)
    assertEqual(net1, net2)

def test_NetForward1() -> None:
    net = Network()
    net.default((3, 2, 2), init="test")
    net.brain[-1].activation = "identity"
    input = [1, 2, 3]
    net.forward(input)
    assertEqual(len(net.output), 2)
    assertListEqual(net.output, [15, 15])

def test_NetForward2() -> None:
    net = Network()
    net.default((4, 4, 4), init="test")
    layer1 = Layer()
    layer1.straight(4, "clip", [-12, 150], "test")
    net.modify_layer(layer1, 0)
    layer2 = Layer()
    layer2.straight(4, "clip", [4, 150], "test")
    net.modify_layer(layer2, -1)
    net.forward([1, 2, 3, 4])
    assertListEqual(net.output, [4, 4, 5, 6])
