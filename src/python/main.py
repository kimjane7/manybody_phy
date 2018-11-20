from pairing_model import PairingModel
from solver import Solver

def main():

	holes = [0,1,2,3]
	particles = [4,5,6,7]
	ds_list = [0.1, 0.5]

	system = PairingModel(1.0,0.5,holes,particles)

	'''
	for ds in ds_list:
		solver = Solver(system, 10.0, ds, euler_option=True)
		solver.SRG("data/srg_euler_"+str(ds)+"_flow.dat")
		del solver

	for ds in ds_list:
		solver = Solver(system, 10.0, ds, euler_option=True)
		solver.SRG_MAGNUS("data/srg_magnus_euler_"+str(ds)+"_flow.dat")
		del solver	

	for ds in ds_list:
		solver = Solver(system, 10.0, ds, euler_option=True)
		solver.IMSRG("data/imsrg_euler_"+str(ds)+"_flow.dat","white")
		del solver

	for ds in ds_list:
		solver = Solver(system, 10.0, ds, euler_option=True)
		solver.IMSRG_MAGNUS("data/imsrg_magnus_euler_"+str(ds)+"_flow.dat","white")
		del solver
	'''


if __name__ == "__main__":
	main()