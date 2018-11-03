// diagonalizes a matrix H using standard SRG and SRG w/ magnus expansion

#include "srg.h"

using namespace std;
using namespace arma;

CSRG::CSRG(mat H0, int N, int Bmax, double smax, double ds){

	H0_ = H0;
	N_ = N;
	Bmax_ = Bmax;
	smax_ = smax;
	ds_ = ds;
	get_prefactors();
}

double CSRG::factorial(int n){

	if(n > 0){
		double logfactorial = 0.0;
		for(int i = 1; i <= n; ++i) logfactorial += log((double) i);
		return exp(logfactorial);
	}
	else return 1.0;
}

double CSRG::binomial_coeff(int n, int k){

	return factorial(n)/(factorial(k)*factorial(n-k));
}

double CSRG::offdiag_H(){

	double sum = 0.0;

	// sum of squares of elements above main diagonal
	for(int i = 0; i < N_; ++i){
		for(int j = i+1; j < N_; ++j){
			sum += H_(i,j)*H_(i,j);
		}
	}

	// double sum and take square root
	return sqrt(2.0*sum);
}

double CSRG::frobenius_norm(mat A){

	double sum = 0.0;

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < N_; ++j){
			sum += A(i,j)*A(i,j);
		}
	}

	return sqrt(sum);
}

void CSRG::get_prefactors(){

	prefactor_.set_size(Bmax_);
	vec B(Bmax_);
	B(0) = 1.0;

	// set all odd Bernoulli numbers
	B(1) = -0.5; for(int n = 3; n < Bmax_; ++n) B(n) = 0.0;

	// use recursive relation to calculate even Bernoulli numbers
	for(int n = 2; n < Bmax_; n += 2){
		B(n) = 0.0;
		for(int k = 0; k < n; ++k) B(n) -= binomial_coeff(n+1,k)*B(k)/(n+1);
	}

	// calculate prefactors
	for(int n = 0; n < Bmax_; ++n) prefactor_(n) = B(n)/factorial(n);
}

void CSRG::display_H(){

	for(int i = 0; i < N_; i++){
		cout << "[";
		for(int j = 0; j < N_; j++){
			cout << setw(10) << setprecision(3) << H_(i,j);
		}
		cout << "]" << endl;
	}
	cout << endl;	
}

mat CSRG::commutator(mat A, mat B){
	
	return A*B-B*A;
}

mat CSRG::nested_commutator(mat A, mat B, int n){

	if(n == 0) return B;
	else{
		mat ad = B;
		for(int i = 0; i < n; ++i) ad = commutator(A,ad);
		return ad;
	}
}

void CSRG::get_eta(){

	// split H into diag and off-diag parts
	mat Hd = diagmat(H_);
	mat Hod = H_-Hd;
	eta_ = commutator(Hd,Hod);		
}

mat CSRG::dH_ds(){

	get_eta();
	return commutator(eta_,H_);
}

mat CSRG::dOmega_ds(){

	int n = 0;
	get_eta();

	// only one non-zero odd Bernoulli number
	mat summand, sum = -0.5*nested_commutator(Omega_,eta_,1);

	// even 
	do{
		summand = prefactor_(n)*nested_commutator(Omega_,eta_,n);
		sum += summand;
		n += 2;
	}while(n < Bmax_);

	return sum;
}

void CSRG::RK4_srg(){

	mat k1, k2, k3, k4, H;

	// store current H
	H = H_;

	// RK4 algorithm
	k1 = ds_*dH_ds(); 

	H_ = H+0.5*k1; 
	k2 = ds_*dH_ds(); 

	H_ = H+0.5*k2; 
	k3 = ds_*dH_ds();

	H_ = H+k3;
	k4 = ds_*dH_ds();

	H_ = H+(k1+2.0*k2+2.0*k3+k4)/6.0;
}

void CSRG::RK4_magnus(){

	mat k1, k2, k3, k4, H, Omega;

	// store current H and Omega
	H = H_;
	Omega = Omega_;

	// RK4 algorithm
	k1 = ds_*dOmega_ds(); 

	Omega_ = Omega+0.5*k1; 
	k2 = ds_*dOmega_ds();  

	Omega_ = Omega+0.5*k2; 
	k3 = ds_*dOmega_ds(); 

	Omega_ = Omega+k3;
	k4 = ds_*dOmega_ds(); 

	Omega_ = Omega+(k1+2.0*k2+2.0*k3+k4)/6.0;
}

void CSRG::srg(vec snapshots, string filename){

	int k = 0;
	H_ = H0_;

	// open file
	ofstream outfile;
	outfile.open(filename);
	outfile << "# dimension of H = " << N_ << endl;
	outfile << "# flow parameter s, norm of off-diagonal elements of H, vectorized elements of H (N*N elements)" << endl;

	for(double s = 0; s <= smax_; s += ds_){

		RK4_srg();

		// write snapshots to file
		if(fabs(s-snapshots(k)) < 0.5*ds_){

			outfile << s << "\t" << offdiag_H() << "\t";

			for(int i = 0; i < N_; ++i){
				for(int j = 0; j < N_; ++j){
					outfile << H_(i,j) << "\t";
				}
			}

			outfile << endl;
			++k;
		}
	}

	outfile.close();
}

void CSRG::magnus(vec snapshots, string filename){

	//int k = 0;
	H_ = H0_;
	Omega_.zeros(N_,N_);

	// open file
	ofstream outfile;
	outfile.open(filename);
	outfile << "# dimension of H = " << N_ << endl;
	outfile << "# flow parameter s, norm of off-diagonal elements of H, vectorized elements of H (N*N elements)" << endl;

	cout << frobenius_norm(Omega_) << endl;
	for(double s = 0; s <= smax_; s += ds_){

		cout << s << "\t" << frobenius_norm(Omega_) << endl;
		outfile << s << "\t" << offdiag_H() << "\t" << frobenius_norm(Omega_) << endl;

		RK4_magnus();

		/*
		// write snapshots to file
		if(fabs(s-snapshots(k)) < 0.5*ds_){

			H_ = expmat(Omega_)*H0_*expmat(-Omega_);

			outfile << s << "\t" << offdiag_H() << "\t" << frobenius_norm(Omega_) << "\t";

			for(int i = 0; i < N_; ++i){
				for(int j = 0; j < N_; ++j){
					outfile << H_(i,j) << "\t";
				}
			}

			outfile << endl;
			++k;
		}
		*/
	}

	outfile.close();
}

mat CSRG::srg(){

	H_ = H0_;
	for(double s = 0; s <= smax_; s += ds_) RK4_srg();
	return H_;
}

mat CSRG::magnus(){

	H_ = H0_;
	Omega_.zeros(N_,N_);

	for(double s = 0; s <= smax_; s += ds_) RK4_magnus();
	return H_;
}

int main(int argc, char *argv[]){

	double d = atof(argv[1]);
	double g = atof(argv[2]);
	double smax = atof(argv[3]);
	double ds = atof(argv[4]);

	// set up Hamiltonian
	mat H0 = zeros<mat>(6,6);
	for(int i = 0; i < 6; ++i){
		for(int j = 0; j < 6; ++j){
			if(i+j != 5) H0(i,j) = -0.5*g; 
		}
	}
	H0(0,0) = 2*d-g;
	H0(1,1) = 4*d-g;
	H0(2,2) = 6*d-g;
	H0(3,3) = 6*d-g;
	H0(4,4) = 8*d-g;
	H0(5,5) = 10*d-g;

	for(int i = 0; i < 6; ++i){
		for(int j = 0; j < 6; ++j){
			cout << H0(i,j) << "\t";
		}
		cout << endl;
	}

	CSRG pairing_model(H0, 6, 170, smax, ds);
	vec snapshots = linspace<vec>(0.0, smax);
	pairing_model.srg(snapshots, "pairing_srg.dat");
	pairing_model.magnus(snapshots, "pairing_srg_magnus.dat");

	return 0;
}