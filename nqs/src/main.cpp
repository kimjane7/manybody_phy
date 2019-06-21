#include "hamiltonian/hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "sampler/metropolis/importancesampling/importancesampling.h"

using namespace std;
using namespace Eigen;

int main(){

	// NQS parameters
	int n_particles = 2;
	int n_hidden = 8;
	int dimension = 2;
	double sigma = 1.0;

	// Hamiltonian parameters
	bool coulomb = true;
	//double hard_core_diameter = 0.0; 
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
	double learning_rate = 0.1;

	// Sampler parameters
	random_device rd;
	int n_cycles = 10;
	int n_samples = 100000;
	double timestep = 0.01;
	string filename = "test.dat";
	string block_filename = "block_test.dat";
	
	

	// main program
	NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
	Hamiltonian H(coulomb, omega, NQS);
	StochasticGradientDescent Optimizer(n_params, learning_rate);
	MetropolisImportanceSampling Sampler(rd(), n_cycles, n_samples, timestep, NQS, H, Optimizer, filename, block_filename);

	Sampler.optimize();

	return 0;
} 