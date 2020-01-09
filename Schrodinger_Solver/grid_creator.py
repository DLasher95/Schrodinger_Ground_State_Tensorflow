import numpy as np

"""
Class for creating a 2D grid

Author: Dylan Lasher
"""

class grid (object):
    def __init__(self, limit = 20, L = 256):
        x = np.linspace(-limit, limit, L)
        y = np.linspace(-limit, limit, L)

        self.x = x
        self.y = y
        self.limit = limit
        self.L = L
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.mesh = np.meshgrid(x, y) # Make the grid
