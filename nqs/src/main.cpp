#include "hamiltonian/hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "sampler/metropolis/importancesampling/importancesampling.h"

using namespace std;
using namespace Eigen;

int main(){

	// NQS parameters
	int n_particles = 1;
	int n_hidden = 4;
	int dimension = 1;
	double sigma = 1.0;

	// Hamiltonian parameters
	bool coulomb = false;
	//double hard_core_diameter = 0.0; 
	VectorXd omega = VectorXd::Ones(dimension);

	// SGD parameters
	int n_params = n_particles*dimension + n_hidden + n_particles*dimension*n_hidden;
	double learning_rate = 0.1;

	// Sampler parameters
	random_device rd;
	int n_cycles = 10;
	int n_samples = 10000000;
	double timestep = 0.01;
	string filename = "test.dat";
	
	

	// main program
	NeuralQuantumState NQS(n_particles, n_hidden, dimension, sigma);
	Hamiltonian H(coulomb, omega, NQS);
	StochasticGradientDescent Optimizer(n_params, learning_rate);
	MetropolisImportanceSampling Sampler(rd(), n_cycles, n_samples, timestep, NQS, H, Optimizer, filename);

	Sampler.optimize();

	return 0;
} 