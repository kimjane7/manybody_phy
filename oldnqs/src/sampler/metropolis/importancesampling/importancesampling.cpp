#include "importancesampling.h"


MetropolisImportanceSampling::MetropolisImportanceSampling(int seed, int n_samples, 
    double tolerance, double timestep, NeuralQuantumState &NQS, 
    Hamiltonian &H, Optimizer &O, string filename):
    Metropolis(seed, n_samples, tolerance, NQS, H, O, filename){

    diffusion_ = 0.5;
    timestep_ = timestep;
    norm01_ = normal_distribution<double>(0.0,1.0);
}


void MetropolisImportanceSampling::get_trial_sample(){

    int i;
    double rand;

    p_ = random_particle_index_(random_engine_);
    trialx_ = NQS_.x_;
    qforce_ = H_.calc_quantum_force(p_, NQS_.x_);

    for(int d = 0; d < NQS_.D_; ++d){

        i = p_*NQS_.D_+d;
        rand = norm01_(random_engine_);
        trialx_(i) += diffusion_*timestep_*qforce_(d) + rand*sqrt(timestep_);
    }
}


double MetropolisImportanceSampling::proposal_ratio(){

    int i;
    double greens = 0.0;

    // calculate trial quantum force for pth particle only
    trial_qforce_ = H_.calc_quantum_force(p_, trialx_);

    // calculate proposal ratio for pth particle
    for(int d = 0; d < NQS_.D_; ++d){

        i = p_*NQS_.D_+d;
        greens += 0.5*(NQS_.x_(i)-trialx_(i))*(trial_qforce_(d)+qforce_(d));
        greens += 0.25*diffusion_*timestep_*(qforce_(d)*qforce_(d)-trial_qforce_(d)*trial_qforce_(d));
    }

    return exp(greens);
}