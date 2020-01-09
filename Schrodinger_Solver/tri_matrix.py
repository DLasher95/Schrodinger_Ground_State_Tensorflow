import scipy.sparse as sp

"""
Construct tri-diagonal matrix with fringes. Break down sparse matrix.

Author: Dylan Lasher
"""

class matrix (object):
    def __init__(self, L, dx, dy):
        self.L = L
        self.dx = dx
        self.dy = dy

        block = sp.diags([-1, 4, -1], [-1, 0, 1], (L, L))  # Main tri-diagonal
        dia = sp.block_diag((block,) * L)  # Create main block diagnonal by repeating
        sup = sp.diags([-1], [L], (L ** 2, L ** 2))  # Super-diagonal fringe
        sub = sp.diags([-1], [-L], (L ** 2, L ** 2))  # Sub-diagonal fringe

        self.T = (dia + sup + sub) / (2*dx*dy)