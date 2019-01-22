// diagonalizes a matrix H using standard SRG and SRG w/ magnus expansion

#ifndef SRG_H
#define SRG_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <armadillo>
#include "time.h"

using namespace std;
using namespace arma;

class CSRG{

public:

	CSRG(mat H0, int N, int Bmax, double smax, double ds);
	~CSRG(){}

	int N_, Bmax_;
	double smax_, ds_;
	vec prefactor_;
	mat H0_, H_, eta_, Omega_;

	double factorial(int n);
	double binomial_coeff(int n, int k);
	double offdiag_H();
	double frobenius_norm(mat A);

	void get_prefactors();
	void display_H();

	mat commutator(mat A, mat B);
	mat nested_commutator(mat A, mat B, int n);

	void get_eta();
	mat dH_ds();
	mat dOmega_ds();

	void RK4_srg();
	void RK4_magnus();

	void srg(vec snapshots, string filename);
	void magnus(vec snapshots, string filename);

	mat srg();
	mat magnus();
};

#endif
