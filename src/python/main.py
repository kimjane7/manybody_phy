from pairing_model import PairingModel
from solver import Solver

def main():

	holes = [0,1,2,3]
	particles = [4,5,6,7]

	system = PairingModel(1.0,0.5,holes,particles)
	solver = Solver(system, 0.01, 10.0)
	solver.SRG("srg_flow.dat")
	solver.SRG_MAGNUS("srg_magnus_flow.dat")
	solver.IMSRG("imsrg_flow.dat","white")
	solver.IMSRG_MAGNUS("imsrg_magnus_flow.dat","white")


if __name__ == "__main__":
	main()