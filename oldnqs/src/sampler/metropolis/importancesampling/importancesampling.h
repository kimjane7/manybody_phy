#ifndef IMPORTANCESAMPLING_H
#define IMPORTANCESAMPLING_H

#include "../metropolis.h"

using namespace std;


class MetropolisImportanceSampling : public Metropolis {

private:

	double diffusion_, timestep_;
	normal_distribution<double> norm01_;
	VectorXd qforce_, trial_qforce_;

public:

    MetropolisImportanceSampling(int seed, int n_samples, double tolerance, double timestep, 
    	                         NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O, 
								 string filename);
    ~MetropolisImportanceSampling(){}

    void get_trial_sample();
    double proposal_ratio();
};

#endif