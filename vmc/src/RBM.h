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

class CRBM{

public:

	int M_, N_;
	double sigma2_;
	ivec h_;
	vec x_, a_, b_; 
	mat W_;

	CRBM(int number_hidden, double sigma, vec input);
	~CRBM(){}

	double energy();

};

#endif