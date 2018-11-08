import numpy as np
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode
import sys


class IMSRG:

	def __init__(self, d, g, smax, ds, holes, particles, generator):

		self.d = d
		self.g = g
		self.smax = smax
		self.ds = ds
		self.holes = holes
		self.parts = particles
		self.dim1B = len(holes)+len(particles)
		self.dim2B = self.dim1B**2
		self.tolerance = 10e-8

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

		# choice of generator
		if (generator == "wegner") or (generator == "white"):
			self.generator = generator
		else:
			print("Please choose a generator.")
			sys.exit(0)


		# magnus expansion coefficients
		self.max = 100
		self.build_prefactors()


	########################
	####### IMSRG(2) #######
	########################


	def build_bas2B(self):

		# construct list of states
		self.bas2B = []

		for i in self.holes:
			for j in self.holes:
				self.bas2B.append((i,j))

		for i in self.holes:
			for a in self.parts:
				self.bas2B.append((i,a))

		for a in self.parts:
			for i in self.holes:
				self.bas2B.append((a,i))

		for a in self.parts:
			for b in self.parts:
				self.bas2B.append((a,b))


		# store indices of states in dictionary
		self.idx2B = {}

		for i, state in enumerate(self.bas2B):
			self.idx2B[state] = i;


	def build_ph_bas2B(self):

		# construct list of states
		self.ph_bas2B = []

		for i in self.holes:
			for j in self.holes:
				self.ph_bas2B.append((i,j))

		for i in self.holes:
			for a in self.parts:
				self.ph_bas2B.append((i,a))

		for a in self.parts:
			for i in self.holes:
				self.ph_bas2B.append((a,i))

		for a in self.parts:
			for b in self.parts:
				self.ph_bas2B.append((a,b))


		# store indices of states in dictionary
		self.ph_idx2B = {}

		for i, state in enumerate(self.ph_bas2B):
			self.ph_idx2B[state] = i;


	def build_occ1B(self):

		self.occ1B = np.zeros(self.dim1B)

		for i in self.holes:
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
			H1B[i,i] = self.d*np.floor_divide(i,2)

		for (i,j) in self.bas2B:
			if (i%2==0 and j==i+1):
				for (k,l) in self.bas2B:
					if (k%2==0 and l==k+1):
						H2B[self.idx2B[(i,j)],self.idx2B[(k,l)]] = -0.5*self.g
						H2B[self.idx2B[(i,j)],self.idx2B[(l,k)]] = 0.5*self.g
						H2B[self.idx2B[(j,i)],self.idx2B[(k,l)]] = 0.5*self.g
						H2B[self.idx2B[(j,i)],self.idx2B[(l,k)]] = -0.5*self.g


		# normal order hamiltonian
		self.E0 = 0.0
		for i in self.holes:
			self.E0 += H1B[i,i]
			for j in self.holes:
				self.E0 += 0.5*H2B[self.idx2B[(i,j)],self.idx2B[(i,j)]]

		self.f0 = H1B.copy()
		for p in self.bas1B:
			for q in self.bas1B:
				for i in self.holes:
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


	def commutator(self, A, B):

		return dot(A,B)-dot(B,A)


	def calc_eta(self):

		# calculate generator chosen by user
		if(self.generator == "wegner"):
			self.calc_eta_wegner()
		if(self.generator == "white"):
			self.calc_eta_white()


	def calc_eta_wegner(self):

		# split into diag and off-diag parts
		fod = np.zeros_like(self.f)
		for i in self.holes:
			for a in self.parts:
				fod[i,a] = self.f[i,a]
				fod[a,i] = self.f[a,i]
		fd = self.f-fod

		Gammaod = np.zeros_like(self.Gamma)
		for i in self.holes:
			for j in self.holes:
				for a in self.parts:
					for b in self.parts:
						ij = self.idx2B[(i,j)]
						ab = self.idx2B[(a,b)]
						Gammaod[ij,ab] = self.Gamma[ij,ab]
						Gammaod[ab,ij] = self.Gamma[ab,ij]
		Gammad = self.Gamma-Gammaod


		eta0B, self.eta1B, self.eta2B = self.commutator2B(fd,Gammad,fod,Gammaod)


	def calc_eta_white(self):

		# one-body part
		self.eta1B = np.zeros_like(self.f)
		for i in self.holes:
			for a in self.parts:
				ai = self.idx2B[(a,i)]
				value = self.f[a,i]/(self.f[a,a]-self.f[i,i]+self.Gamma[ai,ai])
				self.eta1B[a,i] = value
				self.eta1B[i,a] = -value

		# two-body part
		self.eta2B = np.zeros_like(self.Gamma)
		for i in self.holes:
			for j in self.holes:
				for a in self.parts:
					for b in self.parts:
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

		# print("calculating dE, df, dGamma...")

		self.dE, self.df, self.dGamma = self.commutator2B(self.eta1B,self.eta2B,self.f,self.Gamma)



	def derivative(self,t,y):

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


	def imsrg(self, filename):

		# open file
		outfile = open(filename,"w")
		outfile.write("#   delta = {:<5.3f}, g = {:<5.3f}\n".format(self.d, self.g))
		outfile.write('#   {:<14}{:<14}{:<9}\n\n'.format("s","E","dE"))

		# initial values
		y0 = np.append([self.E0],np.append(reshape(self.f0,-1),reshape(self.Gamma0,-1)))
		self.E, self.f, self.Gamma = self.E0, self.f0, self.Gamma0
		self.calc_eta()
		self.calc_dH()

		# integrate
		solver = ode(self.derivative,jac=None)
		solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
		solver.set_initial_value(y0, 0.0)

		print("="*30)
		print("  {:<10}{:<10}{:<10}".format("s","E","dE/ds"))
		print("="*30)

		while solver.successful() and solver.t < self.smax:
			print("{:<10.5f}{:<10.5f}{:<10.5f}".format(solver.t,self.E,self.dE))
			outfile.write('{:<14.7f}{:<14.7f}{:<14.7f}\n'.format(solver.t,self.E,self.dE))
			ys = solver.integrate(self.smax, step=True)
			solver.integrate(solver.t+self.ds)
			if(abs(self.dE/self.E) < self.tolerance): break

		outfile.close()
		


	########################
	### MAGNUS EXPANSION ###
	########################

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



	def factorial(self,n):

		if(n > 0):
			logfactorial = 0.0
			for k in range(1,n+1):
				logfactorial += np.log(k)
			return np.exp(logfactorial)

		else:
			return 1.0

	
	def binomial_coeff(self,n,k):

		return self.factorial(n)/(self.factorial(k)*self.factorial(n-k))


	def build_prefactors(self):	

		# store factorial values
		self.factorials = []
		for n in range(self.max):
			self.factorials.append(self.factorial(n))

		# calculate Bernoulli numbers
		B = np.zeros(self.max)
		B[0] = 1.0
		B[1] = -0.5
		for n in range(2,self.max,2):
			for k in range(n):
				B[n] -= self.binomial_coeff(n+1,k)*B[k]/(n+1)

		# store prefactors
		self.prefactors = []
		for n in range(self.max):
			self.prefactors.append(B[n]/self.factorials[n])


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
			for i in self.holes:
				for a in self.parts:
					C0B += (A1B[i,a]*B1B[a,i]-A1B[a,i]*B1B[i,a])

			# 2B-2B
			if (sgn == 1.0):
				for i in self.holes:
					for j in self.holes:
						for a in self.parts:
							for b in self.parts:
								ij = self.idx2B[(i,j)]
								ab = self.idx2B[(a,b)]
								C0B += 0.5*A2B[ij,ab]*B2B[ab,ij]


		# one-body part
		# 1B-1B
		C1B = self.commutator(A1B,B1B)

		# 1B-2B
		for p in range(self.dim1B):
			for q in range(self.dim1B):
				for i in self.holes:
					for a in self.parts:
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
				for i in self.holes:
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


	def calc_dOmega(self):

		#print("calculating dOmega...")

		# k=0 term
		self.dOmega1B = self.eta1B.copy()
		self.dOmega2B = self.eta2B.copy()

		# k=1 term (only odd term)
		C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,self.eta1B,self.eta2B)
		self.dOmega1B += self.prefactors[1]*C1B
		self.dOmega2B += self.prefactors[1]*C2B

		# remaining even terms
		k = 2
		while (k < self.max and linalg.norm(C2B) > self.tolerance):

			C0B, C1B, C2B = self.commutator2B(self.Omega1B,self.Omega2B,C1B,C2B)
			if(k%2 == 0):
				self.dOmega1B += self.prefactors[k]*C1B
				self.dOmega2B += self.prefactors[k]*C2B	
			k += 1
		self.kterms = k
		


	def calc_H(self):

		#print("calculating E, f, Gamma...")

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
		


	def fod_norm(self):

		norm = 0.0
		for a in self.parts:
			for i in self.holes:
				norm += self.f[a,i]**2+self.f[i,a]**2

		return np.sqrt(norm)



	def Gammaod_norm(self):

		norm = 0.0
		for a in self.parts:
			for b in self.parts:
				for i in self.holes:
					for j in self.holes:
						norm += self.Gamma[self.idx2B[(a,b)],self.idx2B[(i,j)]]**2 + self.Gamma[self.idx2B[(i,j)],self.idx2B[a,b]]**2

		return np.sqrt(norm)


	'''
	def derivative_magnus(self,t,y):

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

	'''

	def magnus(self, filename):


		# open file
		outfile = open(filename,"w")
		outfile.write("#   delta = {:<5.3f}, g = {:<5.3f}, ds = {:<5.3f}\n".format(self.d, self.g, self.ds))
		outfile.write('#   {:<14}{:<14}{:<9}{:<15}{:<13}\n\n'.format("s","E","dE","||Gammaod||","||eta2B||"))

		# initial Omega
		self.Omega1B = np.zeros((self.dim1B,self.dim1B))
		self.Omega2B = np.zeros((self.dim2B,self.dim2B))
		y0 = np.append(reshape(self.Omega1B,-1),reshape(self.Omega2B,-1))

		# initial hamiltonian
		self.E, self.f, self.Gamma = self.E0.copy(), self.f0.copy(), self.Gamma0.copy()
		self.calc_eta()
		self.calc_dH()
		self.calc_dOmega()

		# forward euler
		s = 0.0
		print("="*50)
		print("  {:<10}{:<9}{:<9}{:<10}{:<10}".format("s","E","dE/ds","||eta||","kterms"))
		print("="*50)
		while s < self.smax:
			print("{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10d}".format(s,self.E,self.dE,linalg.norm(self.eta2B),self.kterms))
			outfile.write('{:<14.7f}{:<14.7f}{:<14.7f}{:<14.7f}{:<14.7f}\n'.format(s,self.E,self.dE,self.Gammaod_norm(),linalg.norm(self.eta2B)))
			self.check_hermiticity()
			self.calc_eta()
			self.calc_dOmega()
			self.Omega1B += self.ds*self.dOmega1B
			self.Omega2B += self.ds*self.dOmega2B
			self.calc_H()
			self.calc_dH()
			s += self.ds

		outfile.close()



########################
##### MAIN PROGRAM #####
########################




def main():

	holes = [0,1,2,3]
	particles = [4,5,6,7]

	PairingModel = IMSRG(1.0,0.5,10.0,0.1,holes,particles,"white")
	#PairingModel.imsrg("imsrg_flow.dat")
	#import pdb; pdb.set_trace()
	PairingModel.magnus("magnus_flow.dat")

if __name__ == "__main__":
	main()
