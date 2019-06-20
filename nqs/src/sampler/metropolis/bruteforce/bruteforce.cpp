#include "bruteforce.h"

MetropolisBruteForce::MetropolisBruteForce(int seed, int n_cycles, int n_samples, double maxstep,
    NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O, string filename, string block_filename):
    Metropolis(seed, n_cycles, n_samples, NQS, H, O, filename, block_filename){

    maxstep_ = maxstep;
    random_step_ = uniform_real_distribution<double>(-1.0,1.0);
}

void MetropolisBruteForce::get_trial_sample(){

    p_ = random_particle_index_(random_engine_);
    trialx_ = NQS_.x_;

    for(int d = 0; d < NQS_.D_; ++d){
        trialx_(p_*NQS_.D_+d) += maxstep_*random_step_(random_engine_);
    }
}

double MetropolisBruteForce::proposal_ratio(){

    return 1.0;
}