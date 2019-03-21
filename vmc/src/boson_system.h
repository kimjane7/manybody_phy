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
	int randint(int max);
	double rand01_();
	double randnorm_();

public:

	int D_, N_, P_, B_, b_;
	double a_, E, E_err_, timestep_, diff_coeff_, psi_, psi_new_;
	vec delta_E_, omega2_, alpha_;
	mat r_, r_new_, qforce_, qforce_new_, batches_;

	CBosonSystem();
	~CBosonSystem(){}
	

	

};

#endif