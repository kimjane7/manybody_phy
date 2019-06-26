#include "metropolis.h"

Metropolis::Metropolis(int seed, int n_samples, double tolerance, NeuralQuantumState &NQS, 
                       Hamiltonian &H, Optimizer &O, string filename):
    Sampler(seed, n_samples, tolerance, NQS, H, O, filename){

    unif01_ = uniform_real_distribution<double>(0.0,1.0);
    random_particle_index_ = uniform_int_distribution<int>(0,NQS_.P_-1);

    trialx_.resize(NQS_.M_);
}

void Metropolis::sample(bool &accepted){

    get_trial_sample();

    double prob_ratio = probability_ratio();
    double prop_ratio = proposal_ratio();
    double acceptance_ratio = prob_ratio*prop_ratio;

    if(unif01_(random_engine_) < acceptance_ratio){
        accepted = true;
        NQS_.x_ = trialx_;
        H_.NQS_.x_ = trialx_;
    }
    else{
        accepted = false;
    }
}

double Metropolis::probability_ratio(){

    double psi = H_.calc_psi(NQS_.x_);
    double trialpsi = H_.calc_psi(trialx_);

    return pow(trialpsi/psi,2.0);
}