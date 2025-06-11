"""___Modules_______________________________________________________________"""

# BrainWaveEngine
from .network import Network

# CUE_Simulation
from ..CUE_Simulation.cards import Card
from ..CUE_Simulation.decks import Deck

"""___Class_________________________________________________________________"""

class Save() :

    def __init__(self, path : str) -> None:
        txt = open(path, "r")
        raw_data = txt.readlines()
        txt.close()
        self.data = []
