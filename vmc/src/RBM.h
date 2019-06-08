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

public:

	int P_, D_, M_, N_;
	ivec h_;
	vec x_, a_, b_, B_, f_, Omega2_; 
	mat W_;

	RBM(int number_particles, int number_hidden, vec omega);
	~RBM(){}

	void steepest_gradient_descent(int number_MC_cycles, double tolerance, string filename);
	void variational_energy(int number_MC_cycles);

	double calc_trial_wavefunction(vec x);
	double calc_local_energy();

	vec calc_quantum_force(vec x);
	vec calc_gradient_wavefunction();
	vec calc_gradient_local_energy();

	void store_factors();
	void random_initial_positions();
	void random_new_position(int i);
	double acceptance_ratio(int i);
	double distance(vec x, int i, int p);

};

#endif