# This program solves the pairing model (with 4 particles in 
# 4 doubly-degenerate sp states and Sz = 0)
# using SRG, SRG w/ Magnus, IM-SRG(2), and IM-SRG(2) w/ Magnus.

import numpy as np
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm
from scipy.integrate import odeint, ode
import sys

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

		print("True ground state energy = ", self.energies[0])


class Solver:

	def __init__(self, pairing_model, ds, smax):

		self.pairing_model = pairing_model
		self.ds = ds
		self.smax = smax
		self.dim1B = len(self.pairing_model.holes)+len(self.pairing_model.parts)
		self.dim2B = self.dim1B**2
		self.tolerance = 10e-8

		# magnus expansion coefficients
		self.max = 100
		#self.precalculate_coeffs()



	#################### COMMON #####################

	def commutator(self, A, B):

		return dot(A,B)-dot(B,A)


	def factorial(self, k):

		if k > 0:
			logfactorial = 0.0
			for i in range(1,k+1):
				logfactorial += np.log(i)
			return np.exp(logfactorial)
		else:
			return 1.0


	def binomial_coeff(self, n, k):

		return self.factorial(n)/(self.factorial(k)*self.factorial(n-k))


	def precalc_coeffs(self):

		# store factorials
		self.factorials = []
		for k in range(self.max):
			self.factorials.append(self.factorial(k))

		# calculate Bernoulli numbers
		B = np.zeros(self.max)
		B[0] = 1.0
		B[1] = -0.5
		for k in range(2,self.max,2):
			for i in range(k):
				B[k] -= self.binomial_coeff(k+1,i)*B[i]/(k+1)

		# store coefficients
		self.coeffs = []
		for k in range(self.max):
			self.coeffs.append(B[k]/self.factorials[k])


	###################### SRG ######################

	def srg_derivative(self, t, y):

		# get hamiltonian from linear array y
		self.H = reshape(y,(6,6))

		# calculate eta wegner
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		eta = self.commutator(self.Hd,self.Hod)

		# calculate dH/ds
		dH = self.commutator(eta, self.H)

		# reshape into linear array
		dy = reshape(dH,-1)

		return dy


	def SRG(self, filename):

		# open file
		outfile = open(filename,"w")
		outfile.write("# delta = {:<5.3f}, g = {:<5.3f}\n".format(self.pairing_model.d, self.pairing_model.g))
		outfile.write("# flow parameter s, ||Hod||, diagonal elements of Hd\n".format(self.pairing_model.d, self.pairing_model.g))

		# initial hamiltonian
		self.H = self.pairing_model.hamiltonian
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		y0 = reshape(self.H,-1)

		# integrate
		solver = ode(self.srg_derivative,jac=None)
		solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
		solver.set_initial_value(y0, 0.0)

		while solver.successful() and solver.t < self.smax:
			outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
				   .format(solver.t,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
			ys = solver.integrate(self.smax, step=True)
			solver.integrate(solver.t+self.ds)
			if(linalg.norm(self.Hod) < self.tolerance): break

		outfile.close()


	################# SRG w/ Magnus #################

	def nested_commutator(self, A, B, k):

		if k == 0:
			return B
		else:
			ad = B
			for i in range(0,k):
				ad = self.commutator(A,ad)
			return ad

	def srg_magnus_derivative(self, t, y):

		# get Omega from linear array y
		self.Omega = reshape(y,(6,6))

		# calculate new H
		self.H = expm(self.Omega)*self.pairing_model.hamiltonian*expm(-self.Omega)

		# calculate eta wegner
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		eta = self.commutator(self.Hd,self.Hod)

		# calculate dOmega/ds
		dOmega = self.coeffs[1]*self.nested_commutator(self.Omega,eta,1)
		for k in range(0,self.max,2):
			summand = self.coeffs[k]*self.nested_commutator(self.Omega,eta,k)
			dOmega += summand
			if linalg.norm(summand) < self.tolerance: break

		# reshape into linear array
		dy = reshape(dOmega,-1)

		print("check")

		return dy


	def SRG_MAGNUS(self, filename):

		# open file
		outfile = open(filename,"w")
		outfile.write("# delta = {:<5.3f}, g = {:<5.3f}\n".format(self.pairing_model.d, self.pairing_model.g))
		outfile.write("# flow parameter s, ||Hod||, diagonal elements of Hd\n".format(self.pairing_model.d, self.pairing_model.g))

		# initial hamiltonian
		self.H = self.pairing_model.hamiltonian
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		
		# initial Omega
		self.Omega = np.zeros((6,6))
		y0 = reshape(self.Omega,-1)

		# integrate
		solver = ode(self.srg_magnus_derivative,jac=None)
		solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
		solver.set_initial_value(y0, 0.0)

		while solver.successful() and solver.t < self.smax:
			outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
				   .format(solver.t,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
			ys = solver.integrate(self.smax, step=True)
			solver.integrate(solver.t+self.ds)
			if linalg.norm(self.Hod) < self.tolerance: break

		outfile.close()



def main():

	holes = [0,1,2,3]
	particles = [4,5,6,7]

	system = PairingModel(1.0,0.5,holes,particles)
	solver = Solver(system, 0.01, 10.0)
	solver.SRG("srg_magnus_flow.dat")


if __name__ == "__main__":
	main()