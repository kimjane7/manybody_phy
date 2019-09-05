#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include "../metropolis.h"


class MetropolisBruteForce : public Metropolis {

private:

	double maxstep_;
	uniform_real_distribution<double> random_step_;

public:

    MetropolisBruteForce(int seed, int n_samples, double tolerance, double maxstep,
                         NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O, 
                         string filename);

    void get_trial_sample();
    double proposal_ratio();
};

#endif