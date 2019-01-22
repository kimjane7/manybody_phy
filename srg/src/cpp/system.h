#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <armadillo>
#include "time.h"

using namespace std;
using namespace arma;

class CFermionSystem{

public:

	int N_, F_;
	int particles_, sp_states_; // doubly-degenerate sp states
	double d_, g_;

	imat states_;

	CFermionSystem();
	CFermionSystem(int particles, int sp_states, double d, double g);
	~CFermionSystem(){}

	void generate_basis();

	double h0(int q, int r);
	double v(int q, int r, int s, int t);
	double f(int q, int r);
	double eps(int q, int r);
	double eps(ivec holes, ivec parts);

};

#endif