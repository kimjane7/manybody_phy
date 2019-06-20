#include "sgd.h"

StochasticGradientDescent::StochasticGradientDescent(int n_params, double learning_rate):
    Optimizer(n_params){

    eta_ = learning_rate;
}

void StochasticGradientDescent::optimize_weights(VectorXd gradient, NeuralQuantumState &NQS){

    int index;

    NQS.a_ -= eta_*gradient(seq(0,NQS.M_-1));
    NQS.b_ -= eta_*gradient(seq(NQS.M_,NQS.M_+NQS.N_-1));

    for(int j = 0; j < NQS.N_; ++j){

        index = NQS.M_+NQS.N_+j*NQS.M_;
        NQS.W_.col(j) -= eta_*gradient(seq(index,index+NQS.M_-1));
    }
}