#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "../neuralquantumstate/neuralquantumstate.h"



class Hamiltonian{

private:

	double a0_;
	VectorXd omega_, omega2_;

public:

	NeuralQuantumState &NQS_;

	Hamiltonian(double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS);
	~Hamiltonian(){}

	double calc_local_energy();
	VectorXd calc_gradient_logpsi();
	VectorXd calc_quantum_force(int p);
	VectorXd calc_quantum_force(int p, VectorXd x);
};

#endif