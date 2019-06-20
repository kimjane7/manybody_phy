#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "../sampler.h"

using namespace std;
using Eigen::VectorXd;



class Metropolis : public Sampler {

private:

	uniform_real_distribution<double> unif01_;

protected:

	int p_;
	VectorXd trialx_;
    uniform_int_distribution<int> random_particle_index_;

public:

	Metropolis(int seed, int n_cycles, int n_samples,
               NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O,
               string filename, string block_filename);
	~Metropolis(){}

	void sample(bool &accepted);
	double probability_ratio();

	virtual void get_trial_sample() = 0;
	virtual double proposal_ratio() = 0;
};

#endif // METROPOLIS_H
