import scipy.sparse as sp
import scipy.sparse.linalg as la

"""
Takes in LxL matrix representing quantum potential and solves
the Time-Independent Schrodinger Equation (TISE)

Author: Dylan Lasher
"""

class tise (object):
    def solve(self, potential, L, T):
        self.potential = potential
        self.L = L
        self.T = T

        V = sp.lil_matrix((L ** 2, L ** 2))  # Empty sparse matrix
        V.setdiag(potential.flatten())  # Flatten potential and write along diagonal of V matrix

        H = T + V  # Define Hamiltonian

        # eigs() - Routine to find eigenvalues of Hamiltonian
        # Request 5 smallest (SM) eigenvalues and their corresponding
        # eigenvectors (return_eigenvectors = True).
        # Need smallest now, but we want the more excited states for later
        E, psi = la.eigs(H, k=5, which='SM', return_eigenvectors=True)

        return E, psi