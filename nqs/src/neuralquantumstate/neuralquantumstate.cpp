#include "neuralquantumstate.h"

NeuralQuantumState::NeuralQuantumState(int n_particles, int n_hidden, int dimension, double sigma){
    
    random_device rd;
    random_engine_ = mt19937_64(rd());
    setup(n_particles, n_hidden, dimension, sigma);
}

NeuralQuantumState::NeuralQuantumState(int n_particles, int n_hidden, int dimension, int seed, double sigma){
    
    random_engine_ = mt19937_64(seed);
    setup(n_particles, n_hidden, dimension, sigma);
}

void NeuralQuantumState::setup(int n_particles, int n_hidden, int dimension, double sigma){

    P_ = n_particles;
    D_ = dimension;
    M_ = P_*D_;
    N_ = n_hidden;

    sigma_ = sigma;
    sigma2_ = sigma*sigma;

    x_.resize(M_);                // visible nodes/positions of particles
    h_.resize(N_);                // hidden nodes
    a_.resize(M_);                // visible bias
    b_.resize(N_);                // hidden bias
    W_.resize(M_,N_);             // weights

    //init_uniform_positions();
    init_gaussian_positions();
    init_gaussian_weights();
}

void NeuralQuantumState::init_uniform_positions(){

    uniform_real_distribution<double> unif(-0.5,0.5);

    for(int i = 0; i < M_; ++i){

        x_(i) = unif(random_engine_);
    }
}

void NeuralQuantumState::init_gaussian_positions(){

    normal_distribution<double> norm(0.0,sigma_);

    for(int i = 0; i < M_; ++i){

        x_(i) = norm(random_engine_);
    }
}

void NeuralQuantumState::init_gaussian_weights(){

    double sigma_weights = 0.001;
    normal_distribution<double> norm(0.0,sigma_weights);

    for(int i = 0; i < M_; ++i){

        a_(i) = norm(random_engine_);

        for(int j = 0; j < N_; ++j){

            b_(j) = norm(random_engine_);
            W_(i,j) = norm(random_engine_);
        }
    }
}

double NeuralQuantumState::calc_psi(){

    double psi = exp(-0.5*(x_-a_).squaredNorm()/sigma2_);

    VectorXd B = calc_B();
    for(int j = 0; j < N_; ++j){
        psi *= (1.0+exp(B(j)));
    }

    return psi;
}

double NeuralQuantumState::calc_psi(VectorXd x){

    double psi = exp(-0.5*(x-a_).squaredNorm()/sigma2_);

    VectorXd B = calc_B(x);
    for(int j = 0; j < N_; ++j){
        psi *= (1.0+exp(B(j)));
    }

    return psi;
}

double NeuralQuantumState::distance(int p, int q){

    double R2 = 0.0;

    for(int d = 0; d < D_; ++d){
        R2 += pow(x_(D_*p+d)-x_(D_*q+d),2.0);
    }

    return sqrt(R2);
}

double NeuralQuantumState::distance(VectorXd x, int p, int q){

    double R2 = 0.0;

    for(int d = 0; d < D_; ++d){
        R2 += pow(x(D_*p+d)-x(D_*q+d),2.0);
    }

    return sqrt(R2);
}

VectorXd NeuralQuantumState::calc_B(){

    return b_ + (x_.transpose()*W_).transpose()/sigma2_;
}

VectorXd NeuralQuantumState::calc_B(VectorXd x){

    return b_ + (x.transpose()*W_).transpose()/sigma2_;
}

VectorXd NeuralQuantumState::calc_sigmoidB(VectorXd B){

    VectorXd sigmoidB(N_);

    for(int j = 0; j < N_; ++j){

        sigmoidB(j) = 1.0/(exp(-B(j))+1.0);
    }

    return sigmoidB;
}