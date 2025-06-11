# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:15:20 2023

@author: Ernest
"""

"""___Modules_______________________________________________________________"""

# Python
import numpy as np
import matplotlib.pyplot as plt
import copy
from time import perf_counter as clock

# BrainWaveEngine
from .layer import Layer
from .activation import Activation
from .loss import Loss
from .network import Network
from .optimizer import Optimizer

"""___Class_________________________________________________________________"""

net = Network()
net.default([5, 2, 7, 5])

lay = Layer()
lay.dense(7, 5, activation = 'baffwill_v1')

net.modify_layer(lay, 2)
net.forward([1, 5, 6, 12, 18])

start = -2
end = 6


net = Network()
net.default([1, 5, 5, 1])
net.test(-10, 10)


# net.fct(start, end, 1)
# X = np.linspace(start, end, 1000)
# net.forward(X)
# plt.close(fig=0)
# plt.figure(0)
# plt.plot(X, net.output, '-b')
# plt.show()

# print('c =', net.c, 'et d =', net.d, '\n')
# print('w =', net.w, '\n\nb =', net.b)
