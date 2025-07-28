"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .asserts import *
from ..ImportBWE import *

"""___Function______________________________________________________________"""

def test_network_init():
    network = Network()
    assertEqual(network.brain, [])
