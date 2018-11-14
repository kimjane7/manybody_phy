import numpy as np
from numpy import linalg
from scipy.linalg import eigvalsh

class PairingModel:

	def __init__(self, d, g, holes, particles):

		self.d = d
		self.g = g
		self.holes = holes
		self.parts = particles

		# pairing model hamiltonian in mb state basis
		self.hamiltonian = np.zeros((6,6))
		for i in range(0,6):
			for j in range(0,6):
				if (i+j) != 5:
					self.hamiltonian[i,j] = -0.5*g;
		self.hamiltonian[0,0] = 2.0*d-g
		self.hamiltonian[1,1] = 4.0*d-g
		self.hamiltonian[2,2] = 6.0*d-g
		self.hamiltonian[3,3] = 6.0*d-g
		self.hamiltonian[4,4] = 8.0*d-g
		self.hamiltonian[5,5] = 10.0*d-g

		# store exact eigenvalues
		self.energies = linalg.eigvalsh(self.hamiltonian)

		#print('\nTrue ground state energy = ', self.energies[0], '\n')

