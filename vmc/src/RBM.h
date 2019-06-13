#ifndef RBM_H
#define RBM_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

class RBM{

private:

	mt19937 rng_;
	double randnorm();
	double rand01();

public:

	int P_, D_, M_, N_;
	double a0_, timestep_, diff_coeff_;
	double E_, EL_, E_err_, psi_, psi_new_;

	ivec h_;
	vec x_, x_new_, qforce_, qforce_new_, delta_E_;
	vec a_, b_, theta_, B_, f_, Omega2_; 
	mat W_;

	RBM(int number_particles, int number_hidden, double hard_core_diameter, vec omega);
	~RBM(){}

	void steepest_gradient_descent(int number_MC_cycles, double tolerance, string filename);
	void variational_energy(int number_MC_cycles);

	double calc_trial_wavefunction(vec x);
	vec calc_quantum_force(vec x);

	double calc_local_energy();
	vec calc_gradient_wavefunction();

	void random_new_position(int k);
	double acceptance_ratio(int k);
	double distance(mat x, int p, int k);
	void store_factors(vec x);
	void vectorize(vec a, vec b, mat W, vec& theta);
	void split(vec theta, vec& a, vec& b, mat& W);
	void set_initial_nodes();
	void set_initial_params();

};

#endif