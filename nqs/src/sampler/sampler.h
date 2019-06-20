#ifndef SAMPLER_H
#define SAMPLER_H

#include <iostream>
#include "../optimizer/optimizer.h"
#include "../hamiltonian/hamiltonian.h"

using std::mt19937_64;
using std::string;

class Sampler{

protected:

    mt19937_64 random_engine_;
    int n_cycles_, n_samples_;

public:

    NeuralQuantumState NQS_;
    Hamiltonian H_;
    Optimizer &O_;

    Sampler(int seed, int n_cycles, int n_samples,
            NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O,
            string filename, string block_filename);
    ~Sampler(){};

    void optimize();

    virtual void sample(bool &accepted) = 0;
};

#endif