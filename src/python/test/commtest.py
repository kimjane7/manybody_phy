import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from contextlib import redirect_stdout

import numpy as np
with open(os.devnull, 'w') as devnull:
    with redirect_stdout(devnull):
        from pairing_model import PairingModel
        from solver import Solver

DIM = 8
ATOL = 1e-8
print("ATOL=", ATOL, sep='')
print()

holes = [0,1,2,3]
particles = [4,5,6,7]

# Only `holes` and `particles` matter here
imsrg = Solver(PairingModel(1.0, 0.5, holes, particles), 10.0, 0.1, euler_option=True)

A1 = np.random.rand(DIM, DIM)
A2 = np.random.rand(DIM**2, DIM**2)
B1 = np.random.rand(DIM, DIM)
B2 = np.random.rand(DIM**2, DIM**2)

def hermitize(x):
    return (x + np.transpose(x))/2
def antihermitize(x):
    return (x - np.transpose(x))/2
def norm(x):
    return np.sqrt(np.sum(x**2))

A1 = hermitize(A1)
A2 = hermitize(A2)
B1 = hermitize(B1)
B2 = hermitize(B2)

C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
C1_h = hermitize(C1); C2_h = hermitize(C2)
first_failed_test = None
try:
    print('Anti-Hermiticity Test:')
    print('    Norm test:')
    n = abs(C0) 
    print('        ||C0|| =', n, end='')
    assert n < ATOL; print(' < ATOL')
    n = norm(C1_h)
    print('        ||C1 + C1\'||/2 =', n, end='')
    assert n < ATOL; print(' < ATOL')
    n = norm(C2_h)
    print('        ||C2 + C2\'||/2 =', n, end='')
    assert n < ATOL; print(' < ATOL')
except AssertionError as err:
    print(' >= ATOL ***FAILED***')
    if first_failed_test == None:
        first_failed_test = err

try:
    print('    Element-wise test...', end='')
    assert np.all(C1_h < ATOL)
    assert np.all(C2_h < ATOL)
    print(' Passed')
except AssertionError as err:
    print(' ***FAILED***')
    if first_failed_test == None:
        first_failed_test = err

A1[:] = 0.0
A2[:] = 0.0

C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
try:
    print('Zero Test:')
    print('    Norm test:')
    n = abs(C0) 
    print('        ||C0|| =', n, end='')
    assert n < ATOL; print(' < ATOL')
    n = norm(C1)
    print('        ||C1|| =', n, end='')
    assert n < ATOL; print(' < ATOL')
    n = norm(C2)
    print('        ||C2|| =', n, end='')
    assert n < ATOL; print(' < ATOL')
except AssertionError as err:
    print(' >= ATOL ***FAILED***')
    if first_failed_test == None:
        first_failed_test = err

print()
print("First failed test:")
raise first_failed_test
