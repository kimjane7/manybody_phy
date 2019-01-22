// In-medium SRG for the pairing model

#ifndef IMSRG_H
#define IMSRG_H

#include <iostream>
#include <cmath>
#include <string> 
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <armadillo>
#include <map>

using namespace std;
using namespace arma;


// for ordering index map
struct StateComparator{
	bool operator()(const irowvec& left, const irowvec& right) const{
		if(left(0) == right(0)) return left(1) > right(1);
		else return left(0) > right(0);
	}
};


class CIMSRG{

public:

	CIMSRG(int dim1B, double d, double g, double smax, double ds, ivec holes, ivec parts);
	~CIMSRG(){}

	int dim1B_, dim2B_;
	int nholes_, nparts_;

	double d_, g_, E_, dE_;
	double smax_, ds_;

	ivec holes_, parts_;
	ivec basis1B_, occ1B_;

	imat basis2B_, ph_basis2B_;
	imat occ2B_1_, occ2B_2_, occ2B_3_, ph_occ2B_1_;

	mat H1B_, H2B_, eta1B_, eta2B_; 
	mat f_, df_, Gamma_, dGamma_;

	map<irowvec,int,StateComparator> index2B_;
	map<irowvec,int,StateComparator> ph_index2B_;

	void build_basis2B();
	void build_ph_basis2B();
	void build_occ1B();
	void build_occ2B_1();		// n_a - n_b
	void build_occ2B_2();		// 1 - n_a - n_b
	void build_occ2B_3();		// n_a * n_b
	void build_ph_occ2B_1();
	void build_hamiltonian();	// pairing model hamiltonian
	void normal_order();

	double fod_norm();
	double Gammaod_norm();

	mat ph_transform2B(mat matrix2B); 
	mat inverse_ph_transform2B(mat ph_matrix2B);
	mat commutator(mat A, mat B);

	void calc_eta_wegner();
	void calc_derivatives();

	void RK2_imsrg();
	void imsrg(vec snapshots, string filename);
	double imsrg(); // returns E


	int max_;
	double E_i_;
	vec prefactor_, factorial_;
	mat f_i_, Gamma_i_;
	mat Omega0B_, Omega1B_, Omega2B_;
	mat dOmega0B_, dOmega1B_, dOmega2B_;
	// check that Omega0B is zero, then remove


	double factorial(int n);
	double binomial_coeff(int n, int k);
	void build_prefactors();
	void commutator(mat A1B, mat A2B, mat B1B, mat B2B, double& C0B, mat& C1B, mat& C2B);

	void calc_dOmega();
	void calc_hamiltonian();
	void magnus(vec snapshots, string filename);
	double magnus(); // returns E
};

#endif
