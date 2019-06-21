#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "../neuralquantumstate/neuralquantumstate.h"



class Hamiltonian{

private:

	bool coulomb_int_, bosons_;
	double a0_, foo_, EL_;
	VectorXd omega_, omega2_;

	void setup(bool coulomb_int, VectorXd omega);
	void add_coulomb_interaction();
	void add_hardcore_interaction(int i);

public:

	NeuralQuantumState &NQS_;

	Hamiltonian(bool coulomb_int, VectorXd omega, NeuralQuantumState &NQS);
	Hamiltonian(bool coulomb_int, double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS);
	~Hamiltonian(){}

	double calc_local_energy();
	VectorXd calc_gradient_logpsi();
	VectorXd calc_quantum_force(int p);
	VectorXd calc_quantum_force(int p, VectorXd x);
};

#endif