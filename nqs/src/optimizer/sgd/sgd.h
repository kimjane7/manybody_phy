#ifndef SGD_H
#define SGD_H

#include "../optimizer.h"

class StochasticGradientDescent : public Optimizer {
	
public:

	double eta_;

    StochasticGradientDescent(int n_params, double learning_rate);
    ~StochasticGradientDescent(){}

    void optimize_weights(VectorXd gradient, NeuralQuantumState &NQS, Hamiltonian &H);
};

#endif // SGD_H