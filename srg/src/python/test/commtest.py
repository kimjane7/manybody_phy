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
ATOL = 1e-5
print('ATOL = {}'.format(ATOL))
print()

holes = [0,1,2,3]
particles = [4,5,6,7]

RAND_MIN = -10.0
RAND_MAX = 10.0
def rand_array(*dims):
    return (RAND_MAX-RAND_MIN)*np.random.rand(*dims) + RAND_MIN

def hermitize(x):
    return (x + np.transpose(x))/2
def hermitize_upper(x):
    for i in range(0, x.shape[0]):
        for j in range(i+1, x.shape[1]):
            x[j, i] = x[i, j]
    return x
def antihermitize(x):
    return (x - np.transpose(x))/2
def norm(x):
    return np.sqrt(np.sum(x**2))
## These also fail
# def norm(x):
#     return np.max(np.abs(x))
# def norm(x):
#     return np.min(np.abs(x))

def within_atol(val, atol):
    if np.all(np.abs(val) < atol):
        print(' < ATOL')
        return True
    else:
        print(' >= ATOL ***')
        return False

def comm_anti_herm_test(C0, C1, C2, *, atol):
    success = True
    C1_h = hermitize(C1); C2_h = hermitize(C2)

    print('Testing: Anti-Hermiticity:')

    n = abs(C0)
    print('    ||C0|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C1_h)
    print('    ||C1 + C1\'||/2 = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C2_h)
    print('    ||C2 + C2\'||/2 = {}'.format(n), end='')
    success &= within_atol(n, atol)

    if success:
        print("---success---")
    else:
        print("***FAILURE***")
    return success

def comm_zero_test(C0, C1, C2, *, atol):
    success = True

    print('Testing: Zero')

    n = abs(C0)
    print('    ||C0|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C1)
    print('    ||C1|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C2)
    print('    ||C2|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    if success:
        print("---success---")
    else:
        print("***FAILURE***")
    return success

def comm_antisymm_test(C0, C1, C2, D0, D1, D2, *, atol):
    success = True

    print('Testing: Commutator Antisymmetry')
    n = abs(C0 + D0)
    print('    ||C0 + D0|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C1 + D1)
    print('    ||C1 + D1|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    n = norm(C2 + D2)
    print('    ||C2 + D2|| = {}'.format(n), end='')
    success &= within_atol(n, atol)

    if success:
        print("---success---")
    else:
        print("***FAILURE***")
    return success

if __name__ == '__main__':
    success = True

    # Only `holes` and `particles` matter here
    imsrg = Solver(PairingModel(1.0, 0.5, holes, particles), 10.0, 0.1, euler_option=True)

    Z1 = np.zeros((DIM, DIM)); Z2 = np.zeros((DIM**2, DIM**2))
    A1 = rand_array(DIM, DIM); A2 = rand_array(DIM**2, DIM**2)
    B1 = rand_array(DIM, DIM); B2 = rand_array(DIM**2, DIM**2)

    ############################################################################
    ### Antisymmetry ###########################################################
    C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
    D0, D1, D2 = imsrg.commutator2B(B1, B2, A1, A2)
    success &= comm_antisymm_test(C0, C1, C2, D0, D1, D2, atol=ATOL)
    print()

    ############################################################################
    ### Zero ###################################################################
    C0, C1, C2 = imsrg.commutator2B(Z1, Z2, B1, B2)
    success &= comm_zero_test(C0, C1, C2, atol=ATOL)
    print()

    ############################################################################
    ### Anti-Hermiticity #######################################################
    A1_h = hermitize_upper(A1.copy()); A2_h = hermitize_upper(A2.copy())
    B1_h = hermitize_upper(B1.copy()); B2_h = hermitize_upper(B2.copy())

    C0, C1, C2 = imsrg.commutator2B(A1, A2, B1, B2)
    success &= comm_anti_herm_test(C0, C1, C2, atol=ATOL)
    print()
    ############################################################################

    if success:
        print("---All tests passed---")
    else:
        print("***SOME TESTS FAILED***")
