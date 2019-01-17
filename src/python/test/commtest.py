import numpy as np
from ..pairing_model import PairingModel
from ..solver import Solver

DIM = 8
ATOL = 1e-5

holes = [0,1,2,3]
particles = [4,5,6,7]

# Only `holes` and `particles` matter here
imsrg = Solver(PairingModel(1.0, 0.5, holes, particles), 10.0, 0.1, euler_option=True)

A1 = np.random.rand(DIM, DIM)
A2 = np.random.rand(DIM**2, DIM**2)
B1 = np.random.rand(DIM, DIM)
B2 = np.random.rand(DIM**2, DIM**2)

A1 = (A1 + transpose(A1))/2
A2 = (A2 + transpose(A2))/2
B1 = (B1 + transpose(B1))/2
B2 = (B2 + transpose(B2))/2

C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
assert abs(C0) < ATOL
assert np.all(np.abs((C1 + transpose(C1))/2) < ATOL)
assert np.all(np.abs((C2 + transpose(C2))/2) < ATOL)

A1[:] = 0.0
A2[:] = 0.0

C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
assert abs(C0) < ATOL
assert np.all(np.abs(C1) < ATOL)
assert np.all(np.abs(C2) < ATOL)
