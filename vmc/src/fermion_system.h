#ifndef BOSON_SYSTEM_H
#define BOSON_SYSTEM_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

class CFermionSystem{

private:

	mt19937 rng_;
	int randint(int max);
	double rand01();
	double randnorm();

public:

	int D_, N_;
	double a_, E_, E_err_, timestep_, diff_coeff_, psi_, psi_new_;
	vec delta_E_, omega2_, alpha_;
	mat r_, r_new_, qforce_, qforce_new_;

	CFermionSystem(int dimension, int number_fermions, double hard_core_diameter, vec omega);
	~CFermionSystem(){}
	
	void steepest_gradient_descent(int number_MC_cycles, double tolerance, vec alpha0, string filename);
	void variational_energy(int number_MC_cycles);

	double calc_trial_wavefunction(mat r);
	double calc_local_energy();

	mat calc_quantum_force(mat r);

	vec calc_gradient_wavefunction();
	vec calc_gradient_local_energy();

	void random_initial_positions();
	void random_new_position(int i);
	double acceptance_ratio(int i);
	double distance(mat r, int i, int j);

	string alpha_string();

};

#endif