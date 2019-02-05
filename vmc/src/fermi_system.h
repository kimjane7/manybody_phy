#ifndef FERMI_SYSTEM_H
#define FERMI_SYSTEM_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

class CFermiSystem{

private:

	mt19937 rng_;
	double rand01_();

public:

	int dim_, N_, max_;
	double step_, psi_, psi_new_;
	vec alpha_, beta_;
	mat r_, r_new_, E_, E_err_;

	CFermiSystem(int dimension, int number_fermions, int max_variation, double position_step, 
		         double alpha0, double alphaf, double beta0, double betaf);
	~CFermiSystem(){}
	
	void montecarlo_sampling(int number_MC_cycles, string filename);
	void random_initial_positions();
	void random_trial_positions();
	double calc_trial_wavefunction(mat r, double alpha, double beta);
	double calc_local_energy(double alpha, double beta);
	

};

#endif