#include "catch.hpp"
#include "hamiltonian/hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "sampler/metropolis/importancesampling/importancesampling.h"


TEST_CASE("NONINTERACTING PARTICLES IN 1D HARMONIC OSCILLATOR"){

	// NQS parameters
	int n_particles, n_hidden;
	int dimension = 1;
	double sigma = 1.0;

	// Hamiltonian parameters
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params;
	double learning_rate = 0.1;

	// Sampler parameters
	random_device rd;
	int n_samples = 1E5;
	double tolerance = 1E-8;
	double timestep = 0.01;
	string filename;

	SECTION("ONE PARTICLE"){

		n_particles = 1;
		n_hidden = 4;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "1P_1D_importance.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(omega, NQS);
		StochasticGradientDescent Optimizer(n_params, learning_rate);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run one last calculation at minimum
		//Sampler.n_samples_ = 1E7;
		//Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(0.5));
	}

	SECTION("TWO PARTICLES"){

		n_particles = 2;
		n_hidden = 6;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "2P_1D_importance.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(omega, NQS);
		StochasticGradientDescent Optimizer(n_params, learning_rate);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run one last calculation at minimum
		//Sampler.n_samples_ = 1E7;
		//Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(1.0));
	}
}