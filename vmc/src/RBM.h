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
	vec x_, a_, b_, f_, Omega2_; 
	mat W_, B_;

	RBM(int number_particles, int number_hidden, vec omega, vec input);
	~RBM(){}

	double calc_energy();
	double calc_local_energy();

	void store_factors();

};

#endif