#include "catch.hpp"
#include "hamiltonian/hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "sampler/metropolis/bruteforce/bruteforce.h"
#include "sampler/metropolis/importancesampling/importancesampling.h"

TEST_CASE("NONINTERACTING PARTICLES IN 1D HARMONIC OSCILLATOR (BRUTE FORCE)"){

	// NQS parameters
	int n_particles, n_hidden;
	int dimension = 1;
	double sigma = 1.0;

	// Hamiltonian parameters
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params;
	double eta = 0.1;

	// Sampler parameters
	random_device rd;
	int n_samples = 1E5;
	double tolerance = 1E-7;
	double maxstep = 0.1;
	string filename;

	SECTION("ONE PARTICLE"){

		n_particles = 1;
		n_hidden = 4;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "1P_1D_bruteforce.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(omega, NQS);
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisBruteForce Sampler(rd(), n_samples, tolerance, maxstep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(0.5));
	}

	SECTION("TWO PARTICLES"){

		n_particles = 2;
		n_hidden = 8;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "2P_1D_bruteforce.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(omega, NQS);
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisBruteForce Sampler(rd(), n_samples, tolerance, maxstep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(1.0));
	}
}


TEST_CASE("NONINTERACTING PARTICLES IN 1D HARMONIC OSCILLATOR (IMPORTANCE)"){

	// NQS parameters
	int n_particles, n_hidden;
	int dimension = 1;
	double sigma = 1.0;

	// Hamiltonian parameters
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params;
	double eta = 0.1;

	// Sampler parameters
	random_device rd;
	int n_samples = 1E5;
	double tolerance = 1E-7;
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
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(0.5));
	}

	SECTION("TWO PARTICLES"){

		n_particles = 2;
		n_hidden = 8;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "2P_1D_importance.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(omega, NQS);
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples and smaller learning rate
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(1.0));
	}
}


/*
TEST_CASE("ELECTRONS IN 2D HARMONIC OSCILLATOR"){

	// NQS parameters
	int n_particles, n_hidden;
	int dimension = 2;
	double sigma = 1.0;

	// Hamiltonian parameters
	bool electrons = true;
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params;
	double eta = 1E-5;

	// Sampler parameters
	random_device rd;
	int n_samples = 5E6;
	double tolerance = 1E-3;
	double timestep = 0.01;
	string filename;


	SECTION("TWO ELECTRONS"){

		n_particles = 2;
		n_hidden = 8;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "2E_2D_importance.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(electrons, omega, NQS);
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(1.0));
	}
}

TEST_CASE("BOSONS IN 2D HARMONIC OSCILLATOR"){

	// NQS parameters
	int n_particles, n_hidden;
	int dimension = 2;
	double sigma = 1.0;

	// Hamiltonian parameters
	double a0 = 0.1;
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params;
	double eta = 0.1;

	// Sampler parameters
	random_device rd;
	int n_samples = 1E5;
	double tolerance = 1E-3;
	double timestep = 0.1;
	string filename;


	SECTION("TWO BOSONS"){

		n_particles = 2;
		n_hidden = 8;
		n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
		filename = "2B_2D_importance.dat";

		// main program
		NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
		Hamiltonian H(a0, omega, NQS);
		StochasticGradientDescent Optimizer(n_params, eta);
		MetropolisImportanceSampling Sampler(rd(), n_samples, tolerance, timestep, NQS, H, Optimizer, filename);
		Sampler.optimize();

		// run optimization again at minimum with more samples
		Sampler.n_samples_ = 1E7;
		Sampler.optimize();

		// check energy
		REQUIRE(Sampler.EL_mean_ == Approx(1.0));
	}
}

*/