#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "../neuralquantumstate/neuralquantumstate.h"



class Hamiltonian{

private:


	bool electrons_, bosons_;
	double a0_;
	VectorXd omega_, omega2_;

	void setup(VectorXd omega);
	double calc_coulomb_interaction(int p);
	double calc_hardcore_interaction(int p);
	double calc_coulomb_jastrow_factor(VectorXd x);
	double calc_hardcore_jastrow_factor(VectorXd x);

public:

	
	NeuralQuantumState &NQS_;

	Hamiltonian(VectorXd omega, NeuralQuantumState &NQS);
	Hamiltonian(bool electrons, VectorXd omega, NeuralQuantumState &NQS);
	Hamiltonian(double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS);
	~Hamiltonian(){}

	double calc_local_energy();
	double calc_psi(VectorXd x);
	VectorXd calc_gradient_logpsi();
	VectorXd calc_quantum_force(int p, VectorXd x);
};

#endif