#pragma once
#include <random>
#include "../Eigen/Dense"
#include "../definitions.hpp"


class NeuralQuantumState{
public:
    
    int N_, M_;
    double psi_;
    Vector x_, alpha_;
    
    NeuralQuantumState(int n_particles, int n_hidden);
    ~NeuralQuantumState(){}
    
    double distance(Vector x, int p, int q);
    virtual void vectorize_parameters();
    virtual void separate_parameters();
    virtual double calc_psi(Vector x);
    virtual double calc_local_kinetic_energy(x)
}
