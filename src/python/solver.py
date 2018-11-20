import pairing_model
import numpy as np
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm
from scipy.integrate import odeint, ode
import sys


class Solver:

	def __init__(self, pairing_model, smax, ds, euler_option = False):

		self.pairing_model = pairing_model
		self.smax = smax
		self.ds = ds
		self.euler_option = euler_option
		self.dim1B = len(self.pairing_model.holes)+len(self.pairing_model.parts)
		self.dim2B = self.dim1B**2
		self.tolerance = 10e-8

		# magnus expansion coefficients
		self.max = 100
		self.build_coeffs()

		# bases and indices
		self.bas1B = [i for i in range(self.dim1B)]
		self.build_bas2B()
		self.build_ph_bas2B()

		# occupation number matrices
		self.build_occ1B()
		self.build_occ2B()
		self.build_ph_occ2B_A()

		# pairing hamiltonian
		self.normal_order()

		# limit the number of lines written to file
		self.limit = 1000
		self.s_iter = 0
		if (smax/ds) <= self.limit:
			self.s_write = np.arange(0,smax,ds)
			self.limit = len(self.s_write)

		else:
			f = 0.25
			self.s_write = np.append(np.arange(0,f*smax,2.0*f*smax/self.limit),np.arange(f*smax,smax,2*(1-f)*smax/self.limit))

		'''
		print('='*60)
		print('{:^20}{:^20}{:^20}'.format("Method","Ground state energy","Error (%)"))
		print('='*60)
		'''


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
		outfile.write("# "+"="*113+"\n")
		outfile.write('#    {:<11}{:<15}{:<15}\n'.format("s","||Hod||","diagonal elements of H"))
		outfile.write("# "+"="*113+"\n\n")

		# initial hamiltonian
		self.H = self.pairing_model.hamiltonian.copy()
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		y0 = reshape(self.H,-1)

		# store initial "ground state energy"
		E0 = self.Hd[0,0].copy()
		
		if self.euler_option:

			ys = y0.copy()
			s = 0.0

			while s < self.smax:

				# write to file
				if (self.s_iter < self.limit) and (abs(s-self.s_write[self.s_iter]) <= 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(s,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
					self.s_iter += 1

				# euler step
				ys += self.ds*self.srg_derivative(s,ys)
				s += self.ds

				# safe-guards
				if (self.s_iter >= self.limit): break
				if (self.Hd[0,0] > E0):
					print('E > E0 at s = {:8.5f}!'.format(s))
					break
				if (self.Hd[0,0] < 0.0):
					print('E < 0 at s = {:8.5f}'.format(s))
					break


		else:

			solver = ode(self.srg_derivative,jac=None)
			solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
			solver.set_initial_value(y0, 0.0)

			while solver.successful() and solver.t < self.smax:

				# write to file
				if (self.s_iter < self.limit) and (abs(solver.t-self.s_write[self.s_iter]) < 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						.format(solver.t,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
					self.s_iter += 1

				# integrate
				ys = solver.integrate(self.smax, step=True)
				solver.integrate(solver.t+self.ds)

				if(linalg.norm(self.Hod) < self.tolerance): break



		outfile.close()
		self.s_iter = 0

		'''
		error = 100.0*abs(self.Hd[0,0]-self.pairing_model.energies[0])/self.pairing_model.energies[0]
		print('{:^20}{:^20.11}{:^20.5}'.format("SRG",self.Hd[0,0],error))
		'''


	################# SRG w/ Magnus #################

	def srg_magnus_derivative(self, t, y):

		# get Omega from linear array y
		self.Omega = reshape(y,(6,6))


		# calculate new H
		self.H = dot(expm(self.Omega),dot(self.pairing_model.hamiltonian,expm(-self.Omega)))

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


		return dy


	def SRG_MAGNUS(self, filename):

		# open file
		outfile = open(filename,"w")
		outfile.write("# delta = {:<5.3f}, g = {:<5.3f}\n".format(self.pairing_model.d, self.pairing_model.g))
		outfile.write("# "+"="*113+"\n")
		outfile.write('#    {:<11}{:<15}{:<15}\n'.format("s","||Hod||","diagonal elements of H"))
		outfile.write("# "+"="*113+"\n\n")

		# initial hamiltonian
		self.H = self.pairing_model.hamiltonian.copy()
		self.Hd = diag(diag(self.H))
		self.Hod = self.H-self.Hd
		
		# initial Omega
		self.Omega = np.zeros((6,6))
		y0 = reshape(self.Omega,-1)

		# store initial "ground state energy"
		E0 = self.Hd[0,0].copy()
		
		# integrate
		if self.euler_option:

			ys = y0.copy()
			s = 0.0

			while s < self.smax:

				# write to file
				if (self.s_iter < self.limit) and (abs(s-self.s_write[self.s_iter]) <= 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(s,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
					self.s_iter += 1

				# euler step
				ys += self.ds*self.srg_magnus_derivative(s,ys)
				s += self.ds

				# safe-guards
				if (self.s_iter >= self.limit): break
				if (self.Hd[0,0] > E0):
					print('E > E0 at s = {:8.5f}!'.format(s))
					break
				if (self.Hd[0,0] < 0.0):
					print('E < 0 at s = {:8.5f}'.format(s))
					break

		else:

			solver = ode(self.srg_magnus_derivative,jac=None)
			solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
			solver.set_initial_value(y0, 0.0)

			while solver.successful() and solver.t < self.smax:
				if (self.s_iter < len(self.s_write)) and (abs(solver.t-self.s_write[self.si]) < 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(solver.t,linalg.norm(self.Hod),self.Hd[0,0],self.Hd[1,1],self.Hd[2,2],self.Hd[3,3],self.Hd[4,4],self.Hd[5,5]))
					self.s_iter += 1
				ys = solver.integrate(self.smax, step=True)
				solver.integrate(solver.t+self.ds)
				if linalg.norm(self.Hod) < self.tolerance: break

		outfile.close()
		self.si = 0

		'''
		error = 100.0*abs(self.Hd[0,0]-self.pairing_model.energies[0])/self.pairing_model.energies[0]
		print('{:^20}{:^20.11}{:^20.5}'.format("SRG w/ Magnus",self.Hd[0,0],error))

		'''

	
	################### IM-SRG(2) ###################

	def imsrg_derivative(self, t, y):

		# get hamiltonian from linear array y
		ptr = 0
		self.E = y[ptr]

		ptr = 1
		self.f = reshape(y[ptr:ptr+self.dim1B**2],(self.dim1B,self.dim1B))

		ptr += self.dim1B**2
		self.Gamma = reshape(y[ptr:ptr+self.dim2B**2],(self.dim2B,self.dim2B))

		# calculate rhs
		self.calc_eta()
		self.calc_dH()

		# reshape into linear array
		dy = np.append([self.dE],np.append(reshape(self.df,-1),reshape(self.dGamma,-1)))

		return dy


	def IMSRG(self, filename, generator):

		# choice of generator
		if (generator == "wegner") or (generator == "white"):
			self.generator = generator
		else:
			print("Please choose a generator.")
			sys.exit(0)

		# open file
		outfile = open(filename,"w")
		outfile.write("# delta = {:<5.3f}, g = {:<5.3f}, generator = {:<15}\n".format(self.pairing_model.d,self.pairing_model.g,self.generator))
		outfile.write("# "+"="*84+"\n")
		outfile.write('#    {:<15}{:<13}{:<13}{:<15}{:<14}{:<15}\n'.format("s","E","dE/ds","||eta2B||","||fod||","||Gammaod||"))
		outfile.write("# "+"="*84+"\n\n")

		# initial values
		y0 = np.append([self.E0],np.append(reshape(self.f0,-1),reshape(self.Gamma0,-1)))
		self.E, self.f, self.Gamma = self.E0.copy(), self.f0.copy(), self.Gamma0.copy()
		self.calc_eta()
		self.calc_dH()

		
		# integrate
		if self.euler_option:

			ys = y0.copy()
			s = 0.0

			while s < self.smax:

				# write to file
				if (self.s_iter < self.limit) and (abs(s-self.s_write[self.s_iter]) <= 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(s,self.E,self.dE,linalg.norm(self.eta2B),self.fod_norm(),self.Gammaod_norm()))
					self.s_iter += 1

				# euler step
				ys += self.ds*self.imsrg_derivative(s,ys)
				s += self.ds

				# safe-guards
				if (self.s_iter >= self.limit): break
				if (self.E > self.E0):
					print('E > E0 at s = {:8.5f}!'.format(s))
					break
				if (self.E < 0.0):
					print('E < 0 at s = {:8.5f}'.format(s))
					break

		else:

			solver = ode(self.imsrg_derivative,jac=None)
			solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
			solver.set_initial_value(y0, 0.0)

			while solver.successful() and solver.t < self.smax:
				if (self.si < len(self.s_write)) and (abs(solver.t-self.s_write[self.si]) < 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(solver.t,self.E,self.dE,linalg.norm(self.eta2B),self.fod_norm(),self.Gammaod_norm()))
					self.si += 1
				ys = solver.integrate(self.smax, step=True)
				solver.integrate(solver.t+self.ds)
				if(abs(self.dE/self.E) < self.tolerance): break

		outfile.close()
		self.si = 0

		'''
		error = 100.0*abs(self.E-self.pairing_model.energies[0])/self.pairing_model.energies[0]
		print('{:^20}{:^20.11}{:^20.5}'.format("IMSRG",self.E,error))
		'''


	############## IM-SRG(2) w/ Magnus ##############

	def calc_dOmega(self):

		# k=0 term
		self.dOmega1B = self.eta1B.copy()
		self.dOmega2B = self.eta2B.copy()

		# k=1 term (only odd term)
		C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,self.eta1B,self.eta2B)
		self.dOmega1B += self.coeffs[1]*C1B
		self.dOmega2B += self.coeffs[1]*C2B

		# remaining even terms
		k = 2
		while (k < self.max and linalg.norm(C2B) > self.tolerance):

			C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,C1B,C2B)
			if(k%2 == 0):
				self.dOmega1B += self.coeffs[k]*C1B
				self.dOmega2B += self.coeffs[k]*C2B	
			k += 1
		self.kterms = k


	def calc_H(self):

		# k=0 term
		self.E = self.E0.copy()
		self.f = self.f0.copy()
		self.Gamma = self.Gamma0.copy()

		# k=1 term (only odd term)
		C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,self.f,self.Gamma)
		self.E += C0B
		self.f += C1B
		self.Gamma += C2B

		# remaining even terms
		k = 2
		while (k < self.max and linalg.norm(C2B) > self.tolerance):
			C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,C1B,C2B)
			self.E += C0B/self.factorials[k]
			self.f += C1B/self.factorials[k]
			self.Gamma += C2B/self.factorials[k]
			k += 1


	def imsrg_magnus_derivative(self, t, y):

		# get Omega from linear array y
		ptr = 0
		self.Omega1B = reshape(y[ptr:ptr+self.dim1B**2],(self.dim1B,self.dim1B))

		ptr += self.dim1B**2
		self.Omega2B = reshape(y[ptr:ptr+self.dim2B**2],(self.dim2B,self.dim2B))

		# calculate rhs
		self.calc_eta()
		self.calc_dOmega()

		# reshape into linear array
		dy = np.append(reshape(self.dOmega1B,-1),reshape(self.dOmega2B,-1))

		return dy



	def IMSRG_MAGNUS(self, filename, generator):

		# choice of generator
		if (generator == "wegner") or (generator == "white"):
			self.generator = generator
		else:
			print("Please choose a generator.")
			sys.exit(0)

		# open file
		outfile = open(filename,"w")
		outfile.write("# delta = {:<5.3f}, g = {:<5.3f}, generator = {:<15}\n".format(self.pairing_model.d,self.pairing_model.g,self.generator))
		outfile.write("# "+"="*99+"\n")
		outfile.write('#    {:<15}{:<13}{:<13}{:<15}{:<14}{:<15}{:<15}\n'.format("s","E","dE/ds","||eta2B||","||fod||","||Gammaod||","||Omega2B||"))
		outfile.write("# "+"="*99+"\n\n")

		# initial Omega
		self.Omega1B = np.zeros((self.dim1B,self.dim1B))
		self.Omega2B = np.zeros((self.dim2B,self.dim2B))
		y0 = np.append(reshape(self.Omega1B,-1),reshape(self.Omega2B,-1))

		# initial hamiltonian
		self.E, self.f, self.Gamma = self.E0.copy(), self.f0.copy(), self.Gamma0.copy()
		self.calc_eta()
		self.calc_dH()
		self.calc_dOmega()

		# integrate
		if self.euler_option:

			ys = y0.copy()
			s = 0.0

			while s < self.smax:

				# write to file
				if (self.s_iter < self.limit) and (abs(s-self.s_write[self.s_iter]) <= 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(s,self.E,self.dE,linalg.norm(self.eta2B),self.fod_norm(),self.Gammaod_norm(),linalg.norm(self.Omega2B)))
					self.s_iter += 1

				# euler step
				ys += self.ds*self.imsrg_magnus_derivative(s,ys)
				s += self.ds
				self.calc_H()
				self.calc_dH()

				# safe-guards
				if (self.s_iter >= self.limit): break
				if (self.E > self.E0):
					print('E > E0 at s = {:8.5f}!'.format(s))
					break
				if (self.E < 0.0):
					print('E < 0 at s = {:8.5f}'.format(s))
					break

		else:

			solver = ode(self.imsrg_magnus_derivative,jac=None)
			solver.set_integrator('vode', method='bdf', order=2, nsteps=1000)
			solver.set_initial_value(y0, 0.0)

			while solver.successful() and solver.t < self.smax:
				if (self.si < len(self.s_write)) and (abs(solver.t-self.s_write[self.si]) < 0.5*self.ds):
					outfile.write('{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}{:<15.8f}\n' \
						   .format(solver.t,self.E,self.dE,linalg.norm(self.eta2B),self.fod_norm(),self.Gammaod_norm(),linalg.norm(self.Omega2B)))
					self.si += 1
				ys = solver.integrate(self.smax, step=True)
				solver.integrate(solver.t+self.ds)
				self.calc_H()
				self.calc_dH()
				if(abs(self.dE/self.E) < self.tolerance): break

		outfile.close()
		self.si = 0

		'''
		error = 100.0*abs(self.E-self.pairing_model.energies[0])/self.pairing_model.energies[0]
		print('{:^20}{:^20.11}{:^20.5}'.format("IMSRG w/ Magnus",self.E,error))
		'''


	#################### COMMON #####################


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


	def commutator(self, A, B):

		return dot(A,B)-dot(B,A)


	def nested_commutator(self, A, B, k):

		if k == 0:
			return B
		else:
			ad = B
			for i in range(0,k):
				ad = self.commutator(A,ad)
			return ad
	

	def fod_norm(self):

		norm = 0.0
		for a in self.pairing_model.parts:
			for i in self.pairing_model.holes:
				norm += self.f[a,i]**2+self.f[i,a]**2

		return np.sqrt(norm)


	def Gammaod_norm(self):

		norm = 0.0
		for a in self.pairing_model.parts:
			for b in self.pairing_model.parts:
				for i in self.pairing_model.holes:
					for j in self.pairing_model.holes:
						norm += self.Gamma[self.idx2B[(a,b)],self.idx2B[(i,j)]]**2 + self.Gamma[self.idx2B[(i,j)],self.idx2B[a,b]]**2

		return np.sqrt(norm)


	def check_hermiticity(self):

		# check Hamiltonian is Hermitian
		if (np.allclose(self.Gamma,transpose(self.Gamma))) == False:
			print("Oh no! Gamma is not Hermitian!")

		# check eta is anti-Hermitian
		if (np.allclose(self.eta2B,-transpose(self.eta2B))) == False:
			print("Oh no! Eta is not anti-Hermitian!")

		# check Omega is anti-Hermitian
		if (np.allclose(self.Omega2B,-transpose(self.Omega2B))) == False:
			print("Oh no! Omega is not anti-Hermitian!")


	def build_coeffs(self):

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

	
	def build_bas2B(self):

		# construct list of states
		self.bas2B = []

		for i in self.pairing_model.holes:
			for j in self.pairing_model.holes:
				self.bas2B.append((i,j))

		for i in self.pairing_model.holes:
			for a in self.pairing_model.parts:
				self.bas2B.append((i,a))

		for a in self.pairing_model.parts:
			for i in self.pairing_model.holes:
				self.bas2B.append((a,i))

		for a in self.pairing_model.parts:
			for b in self.pairing_model.parts:
				self.bas2B.append((a,b))


		# store indices of states in dictionary
		self.idx2B = {}

		for i, state in enumerate(self.bas2B):
			self.idx2B[state] = i;		


	def build_ph_bas2B(self):

		# construct list of states
		self.ph_bas2B = []

		for i in self.pairing_model.holes:
			for j in self.pairing_model.holes:
				self.ph_bas2B.append((i,j))

		for i in self.pairing_model.holes:
			for a in self.pairing_model.parts:
				self.ph_bas2B.append((i,a))

		for a in self.pairing_model.parts:
			for i in self.pairing_model.holes:
				self.ph_bas2B.append((a,i))

		for a in self.pairing_model.parts:
			for b in self.pairing_model.parts:
				self.ph_bas2B.append((a,b))


		# store indices of states in dictionary
		self.ph_idx2B = {}

		for i, state in enumerate(self.ph_bas2B):
			self.ph_idx2B[state] = i;	


	def build_occ1B(self):

		self.occ1B = np.zeros(self.dim1B)

		for i in self.pairing_model.holes:
			self.occ1B[i] = 1
	

	def build_occ2B(self):

		# n_p - n_q
		self.occ2B_A = np.zeros((self.dim2B,self.dim2B))

		# 1 - n_p - n_q
		self.occ2B_B = np.zeros((self.dim2B,self.dim2B))

		# n_p * n_q
		self.occ2B_C = np.zeros((self.dim2B,self.dim2B))


		for i, (p,q) in enumerate(self.bas2B):
			self.occ2B_A[i,i] = self.occ1B[p]-self.occ1B[q]
			self.occ2B_B[i,i] = 1-self.occ1B[p]-self.occ1B[q]
			self.occ2B_C[i,i] = self.occ1B[p]*self.occ1B[q]


	def build_ph_occ2B_A(self):

		# n_p - n_q
		self.ph_occ2B_A = np.zeros((self.dim2B,self.dim2B))

		for i, (p,q) in enumerate(self.ph_bas2B):
			self.ph_occ2B_A[i,i] = self.occ1B[p]-self.occ1B[q]


	def normal_order(self):

		# construct pairing hamiltonian
		H1B = np.zeros((self.dim1B,self.dim1B))
		H2B = np.zeros((self.dim2B,self.dim2B))

		for i in self.bas1B:
			H1B[i,i] = self.pairing_model.d*np.floor_divide(i,2)

		for (i,j) in self.bas2B:
			if (i%2==0 and j==i+1):
				for (k,l) in self.bas2B:
					if (k%2==0 and l==k+1):
						H2B[self.idx2B[(i,j)],self.idx2B[(k,l)]] = -0.5*self.pairing_model.g
						H2B[self.idx2B[(i,j)],self.idx2B[(l,k)]] = 0.5*self.pairing_model.g
						H2B[self.idx2B[(j,i)],self.idx2B[(k,l)]] = 0.5*self.pairing_model.g
						H2B[self.idx2B[(j,i)],self.idx2B[(l,k)]] = -0.5*self.pairing_model.g


		# normal order hamiltonian
		self.E0 = 0.0
		for i in self.pairing_model.holes:
			self.E0 += H1B[i,i]
			for j in self.pairing_model.holes:
				self.E0 += 0.5*H2B[self.idx2B[(i,j)],self.idx2B[(i,j)]]

		self.f0 = H1B.copy()
		for p in self.bas1B:
			for q in self.bas1B:
				for i in self.pairing_model.holes:
					self.f0[p,q] += H2B[self.idx2B[(p,i)],self.idx2B[(q,i)]]

		self.Gamma0 = H2B.copy()

		self.E, self.f, self.Gamma = self.E0.copy(), self.f0.copy(), self.Gamma0.copy()


	def ph_transform2B(self, matrix2B):

		ph_matrix2B = np.zeros((self.dim2B,self.dim2B))

		for i, (p,q) in enumerate(self.ph_bas2B):
			for j, (r,s) in enumerate(self.ph_bas2B):
				ph_matrix2B[i,j] -= matrix2B[self.idx2B[(p,s)],self.idx2B[(r,q)]]

		return ph_matrix2B


	def inverse_ph_transform2B(self, ph_matrix2B):

		matrix2B = np.zeros((self.dim2B,self.dim2B))

		for i, (p,q) in enumerate(self.ph_bas2B):
			for j, (r,s) in enumerate(self.ph_bas2B):
				matrix2B[i,j] -= ph_matrix2B[self.idx2B[(p,s)],self.idx2B[(r,q)]]

		return matrix2B


	def calc_eta(self):

		# calculate generator chosen by user
		if(self.generator == "wegner"):
			self.calc_eta_wegner()
		if(self.generator == "white"):
			self.calc_eta_white()


	def calc_eta_wegner(self):

		# split into diag and off-diag parts
		fod = np.zeros_like(self.f)
		for i in self.pairing_model.holes:
			for a in self.pairing_model.parts:
				fod[i,a] = self.f[i,a]
				fod[a,i] = self.f[a,i]
		fd = self.f-fod

		Gammaod = np.zeros_like(self.Gamma)
		for i in self.pairing_model.holes:
			for j in self.pairing_model.holes:
				for a in self.pairing_model.parts:
					for b in self.pairing_model.parts:
						ij = self.idx2B[(i,j)]
						ab = self.idx2B[(a,b)]
						Gammaod[ij,ab] = self.Gamma[ij,ab]
						Gammaod[ab,ij] = self.Gamma[ab,ij]
		Gammad = self.Gamma-Gammaod


		eta0B, self.eta1B, self.eta2B = self.commutator2B(fd,Gammad,fod,Gammaod)


	def calc_eta_white(self):

		# one-body part
		self.eta1B = np.zeros_like(self.f)
		for i in self.pairing_model.holes:
			for a in self.pairing_model.parts:
				ai = self.idx2B[(a,i)]
				value = self.f[a,i]/(self.f[a,a]-self.f[i,i]+self.Gamma[ai,ai])
				self.eta1B[a,i] = value
				self.eta1B[i,a] = -value

		# two-body part
		self.eta2B = np.zeros_like(self.Gamma)
		for i in self.pairing_model.holes:
			for j in self.pairing_model.holes:
				for a in self.pairing_model.parts:
					for b in self.pairing_model.parts:
						ij = self.idx2B[(i,j)]
						ab = self.idx2B[(a,b)]
						ai = self.idx2B[(a,i)]
						aj = self.idx2B[(a,j)]
						bi = self.idx2B[(b,i)]
						bj = self.idx2B[(b,j)]
						denom = (self.f[a,a]+self.f[b,b]-self.f[i,i]-self.f[j,j]
							    +self.Gamma[ab,ab]+self.Gamma[ij,ij]
							    -self.Gamma[ai,ai]-self.Gamma[aj,aj]
							    -self.Gamma[bi,bi]-self.Gamma[bj,bj])
						value = self.Gamma[ab,ij]/denom
						self.eta2B[ab,ij] = value
						self.eta2B[ij,ab] = -value


	def calc_dH(self):

		self.dE, self.df, self.dGamma = self.commutator2B(self.eta1B,self.eta2B,self.f,self.Gamma)


	def commutator2B(self, A1B, A2B, B1B, B2B):


		# zero-body part
		C0B = 0.0
		sgn = 1.0

		# check symmetry
		if (np.allclose(A2B,-transpose(A2B)) and np.allclose(B2B,-transpose(B2B))):
			sgn = -1.0

		if (np.allclose(A2B,transpose(A2B)) and np.allclose(B2B,transpose(B2B))):
			sgn = -1.0

		# zero-body part is non-zero if one is symmetric and other is antisymmetric
		if sgn == 1.0:

			# 1B-1B
			for i in self.pairing_model.holes:
				for a in self.pairing_model.parts:
					C0B += (A1B[i,a]*B1B[a,i]-A1B[a,i]*B1B[i,a])

			# 2B-2B
			if (sgn == 1.0):
				for i in self.pairing_model.holes:
					for j in self.pairing_model.holes:
						for a in self.pairing_model.parts:
							for b in self.pairing_model.parts:
								ij = self.idx2B[(i,j)]
								ab = self.idx2B[(a,b)]
								C0B += 0.5*A2B[ij,ab]*B2B[ab,ij]


		# one-body part
		# 1B-1B
		C1B = self.commutator(A1B,B1B)

		# 1B-2B
		for p in range(self.dim1B):
			for q in range(self.dim1B):
				for i in self.pairing_model.holes:
					for a in self.pairing_model.parts:
						ap = self.idx2B[(a,p)]
						iq = self.idx2B[(i,q)]
						ip = self.idx2B[(i,p)]
						aq = self.idx2B[(a,q)]
						C1B[p,q] += (A1B[i,a]*B2B[ap,iq]-B1B[i,a]*A2B[ap,iq]
							        +B1B[a,i]*A2B[ip,aq]-A1B[a,i]*B2B[ip,aq])

		# 2B-2B

		AB = dot(A2B,dot(self.occ2B_B,B2B))
		ABT = transpose(AB)
		for p in range(self.dim1B):
			for q in range(self.dim1B):
				for i in self.pairing_model.holes:
					ip = self.idx2B[(i,p)]
					iq = self.idx2B[(i,q)]
					C1B[p,q] += 0.5*(AB[ip,iq]+sgn*ABT[ip,iq])

		AB = dot(A2B,dot(self.occ2B_C,B2B))
		ABT = transpose(AB)
		for p in range(self.dim1B):
			for q in range(self.dim1B):
				for r in range(self.dim1B):
					rp = self.idx2B[(r,p)]
					rq = self.idx2B[(r,q)]
					C1B[p,q] += 0.5*(AB[rp,rq]+sgn*ABT[rp,rq])


		# two-body part
		C2B = np.zeros((self.dim2B,self.dim2B))

		# 1B-2B
		for p in range(self.dim1B):
			for q in range(self.dim1B):
				for r in range(self.dim1B):
					for s in range(self.dim1B):
						pq = self.idx2B[(p,q)]
						rs = self.idx2B[(r,s)]
						for t in range(self.dim1B):
							tq = self.idx2B[(t,q)]
							tp = self.idx2B[(t,p)]
							ts = self.idx2B[(t,s)]
							tr = self.idx2B[(t,r)]
							C2B[pq,rs] += (A1B[p,t]*B2B[tq,rs]-B1B[p,t]*A2B[tq,rs]
							              -A1B[q,t]*B2B[tp,rs]+B1B[q,t]*A2B[tp,rs]
							              -A1B[t,r]*B2B[pq,ts]+B1B[t,r]*A2B[pq,ts]
							              +A1B[t,s]*B2B[pq,tr]-B1B[t,s]*A2B[pq,tr])

		# 2B-2B
		
		AB = dot(A2B,dot(self.occ2B_B,B2B))
		ABT = transpose(AB)
		C2B += 0.5*(AB+sgn*ABT)

		# transform to particle-hole representation
		ph_A = self.ph_transform2B(A2B)
		ph_B = self.ph_transform2B(B2B)
		ph_AB = dot(ph_A,dot(self.ph_occ2B_A,ph_B))

		# transform back
		AB = self.inverse_ph_transform2B(ph_AB)

		# antisymmetrization
		asymm_AB = np.zeros_like(AB)
		for pq, (p,q) in enumerate(self.bas2B):
			qp = self.idx2B[(q,p)]
			for rs, (r,s) in enumerate(self.bas2B):
				sr = self.idx2B[(s,r)]
				asymm_AB[pq,rs] += (AB[pq,sr]+AB[qp,rs]-AB[pq,rs]-AB[qp,sr])
		AB = asymm_AB
		C2B += AB

		return C0B, C1B, C2B	

