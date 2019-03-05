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

class CBosonSystem{

private:

	mt19937 rng_;
	double rand01_();

public:

	int dim_, N_, max_;
	double hbar_, m_, a_, step_;
	double psi_, psi_new_, omega_xy_, omega_z_;
	vec alpha_, beta_;
	mat r_, r_new_, E_, E_err_;

	CBosonSystem(int number_bosons, int max_variation, double position_step, double mass, double hard_core_diameter,
	             double omega_xy, double omega_z, double alpha0, double alphaf, double beta0, double betaf);
	~CBosonSystem(){}
	
	void montecarlo_sampling(int number_MC_cycles, string filename);
	void random_initial_positions();
	void random_trial_positions();
	double distance(mat r, int i, int j);
	double calc_trial_wavefunction(mat r, double alpha, double beta);
	double calc_local_energy(mat r, double alpha, double beta);
	

};

#endif