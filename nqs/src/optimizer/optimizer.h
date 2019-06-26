#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../hamiltonian/hamiltonian.h"


class Optimizer{

public:

	int n_params_;

    Optimizer(int n_params);
    ~Optimizer(){}

    virtual void optimize_weights(VectorXd gradient, NeuralQuantumState &NQS, Hamiltonian &H) = 0;
};

#endif