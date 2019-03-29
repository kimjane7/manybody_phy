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


class CGradientDescent{

private:


public:

	// CBosonSystem has member variable delta_
	double (CBosonSystem::* gradient_ptr_)(int number_MC_cycles);

	CGradientDescent(vec &variational_params, mat &data, double (*function_to_minimize)(int number_MC_cycles) );
	~CGradientDescent(){}

	void steepest(vec (*gradient_wrt_params)() );


	

	

};

#endif