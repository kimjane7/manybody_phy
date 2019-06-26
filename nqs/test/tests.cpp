#include "catch.hpp"
#include "hamiltonian/hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "sampler/metropolis/importancesampling/importancesampling.h"


TEST_CASE("ONE PARTICLE IN 1D HARMONIC OSCILLATOR","[1P1D]"){

	cout << "TEST: ONE PARTICLE IN 1D HARMONIC OSCILLATOR" << endl;

	// NQS parameters
	int n_particles = 1;
	int n_hidden = 10;
	int dimension = 1;
	double sigma = 1.0;

	// Hamiltonian parameters
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
	double learning_rate = 0.1;

	// Sampler parameters
	random_device rd;
	int n_cycles = 5;
	int n_samples = 10000000;
	double timestep = 0.01;
	string filename = "1P_1D.dat";

	// main program
	NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
	Hamiltonian H(omega, NQS);
	StochasticGradientDescent Optimizer(n_params, learning_rate);
	MetropolisImportanceSampling Sampler(rd(), n_cycles, n_samples, timestep, NQS, H, Optimizer, filename);

	Sampler.optimize();

	// check energy
	REQUIRE(Sampler.EL_mean_ == Approx(0.5));
}