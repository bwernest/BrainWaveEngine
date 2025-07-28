"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from ..engine import *

# Python
import pytest

"""___Function______________________________________________________________"""

def test_layer():
    layer = Layer()
    layer.dense(2, 3)
    assert layer.weights.size == [2, 3]
