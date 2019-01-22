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

	random_device rd_;
	mt19937 generator_(rd_());

public:

	int dim_, N_, max_;
	double step_, psi_, psi_new_;
	vec alpha_, beta_;
	mat r_, r_new_, E_;

	

	CFermiSystem(int dimension, int number_fermions, int max_variation, double alpha0, double alphaf, double beta0, double betaf);
	~CFermiSystem(){}
	
	void montecarlo_sampling(int number_MC_cycles);
	void random_initial_positions();
	mat random_trial_positions();
	double calc_trial_wavefunction(mat r, double alpha, double beta);
	uniform_real_distribution<double> rand01(0.0, 1.0);	



};

#endif