#include "metropolis.h"

Metropolis::Metropolis(int seed, int n_cycles, int n_samples, NeuralQuantumState &NQS, 
                       Hamiltonian &H, Optimizer &O, string filename):
    Sampler(seed, n_cycles, n_samples, NQS, H, O, filename){

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
    }
    else{
        accepted = false;
    }
}

double Metropolis::probability_ratio(){

    double psi = NQS_.calc_psi(NQS_.x_);
    double trialpsi = NQS_.calc_psi(trialx_);

    if(H_.coulomb_int_){
        psi *= H_.calc_coulomb_jastrow_factor(NQS_.x_);
        trialpsi *= H_.calc_coulomb_jastrow_factor(trialx_);
    }

    if(H_.bosons_){
        psi *= H_.calc_hardcore_jastrow_factor(NQS_.x_);
        trialpsi *= H_.calc_hardcore_jastrow_factor(trialx_);
    }

    return pow(trialpsi/psi,2.0);
}