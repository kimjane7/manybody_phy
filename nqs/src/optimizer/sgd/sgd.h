#ifndef SGD_H
#define SGD_H

#include "../optimizer.h"

class StochasticGradientDescent : public Optimizer {
private:

    double eta_;

public:

    StochasticGradientDescent(int n_params, double learning_rate);
    ~StochasticGradientDescent(){}

    void optimize_weights(VectorXd gradient, NeuralQuantumState &NQS);
};

#endif // SGD_H