"""
Calculate the solution to the SImple Harmonic Oscillator (SHO) Equation

Author: Dylan Lasher
"""
class sho_eq:
    # Takes in meshgrid, two spring constants (k), 2 center coordinates (c)
    # Evaluates the paraboloid, returns 2s potential graphs.
    def V_SHO(mesh, kx, ky, cx=0, cy=0):
        (x, y) = mesh
        V = 0.5 * (kx * (x - cx) ** 2 + ky * (y - cy) ** 2)  # Simple Harmonic Oscillator (SHO) equation
        return V
