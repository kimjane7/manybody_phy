#ifndef NEURALQUANTUMSTATE_H
#define NEURALQUANTUMSTATE_H

#include <random>
#include <cmath>
#include "../Eigen/Dense"

using namespace std;
using namespace Eigen;



class NeuralQuantumState{

private:

    mt19937_64 random_engine_;

    void setup(int n_particles, int n_hidden, int dimension, double sigma);
    void init_uniform_positions();
    void init_gaussian_weights();

public:

    int P_, D_, M_, N_;
    double sigma_, sigma2_;

    VectorXd x_, h_, a_, b_;
    MatrixXd W_;

    NeuralQuantumState(int n_particles, int n_hidden, int dimension, double sigma);
    NeuralQuantumState(int n_particles, int n_hidden, int dimension, int seed, double sigma);
    ~NeuralQuantumState(){}

    double calc_psi();
    double calc_psi(VectorXd x);
    double distance(int p, int q);
    double distance(VectorXd x, int p, int q);
    VectorXd calc_B();
    VectorXd calc_B(VectorXd x);
    VectorXd calc_sigmoidB(VectorXd B);
};

#endif
