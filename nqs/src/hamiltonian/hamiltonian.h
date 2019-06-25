#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "../neuralquantumstate/neuralquantumstate.h"



class Hamiltonian{

private:

	double a0_, foo_, EL_;
	VectorXd omega_, omega2_;

	void setup(bool coulomb_int, VectorXd omega);
	void add_coulomb_interaction();
	void add_hardcore_interaction(int i);

public:

	bool electrons_, bosons_;
	NeuralQuantumState &NQS_;

	Hamiltonian(VectorXd omega, NeuralQuantumState &NQS);
	Hamiltonian(bool electrons, VectorXd omega, NeuralQuantumState &NQS);
	Hamiltonian(double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS);
	~Hamiltonian(){}

	double calc_local_energy();
	double calc_psi(VectorXd x);
	double calc_coulomb_jastrow_factor(VectorXd x);
	double calc_hardcore_jastrow_factor(VectorXd x);
	VectorXd calc_gradient_logpsi();
	VectorXd calc_quantum_force(int p, VectorXd x);
};

#endif