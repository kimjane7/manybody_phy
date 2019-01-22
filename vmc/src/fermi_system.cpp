#include "fermi_system.h"

CFermiSystem::CFermiSystem(int dimension, int number_fermions, int max_variation, double position_step, double alpha0, double alphaf, double beta0, double betaf){

	dim_ = dimension;
	N_ = number_fermions;
	max_ = max_variation;
	step_ = position_step;
	psi_ = 0.0;
	psi_new_= 0.0;

	r_.zeros(N_,dim_);
	r_new_.zeros(N_,dim_);
	alpha_.linspace(alpha0,alphaf,max_);
	beta_.linspace(beta0,betaf,max_);
	E_.zeros(max_,max_);

}

void CFermiSystem::montecarlo_sampling(int number_MC_cycles){

	// loop over various parameter values
	for(int a = 0; a < max_; ++a){
		for(int b = 0; b < max_; ++b){

			// randomly position particles
			random_initial_positions();

			// calculate initial wave function
			psi_ = calc_trial_wavefunction(r_,alpha_(a),beta_(b));

			// loop over monte carlo cycles
			for(int m = 0; m < number_MC_cycles; ++m){

				// propose new trial positions
				random_trial_positions();

				// calculate new trial wave function
				psi_new_ = calc_trial_wavefunction(r_new_,alpha_(a),beta_(b));

				// metropolis test
				if(rand01(generator_) < pow(psi_new_/psi_),2.0){
					r_ = r_new_;
					psi_ = psi_new_;

				}

			}

		}
	}



}

void CFermiSystem::random_initial_positions(){

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			r_(i,j) = step_*(rand01(generator_)-0.5);
		}
	}
}

void CFermiSystem::random_trial_positions(){

	r_new_ = zeros<mat>(max_,max_);

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			r_new_(i,j) = r_(i,j)+step_*(rand01(generator_)-0.5);
		}
	}
}


double CFermiSystem::calc_trial_wavefunction(double r, double alpha, double beta){

	// slater determinant
	double R2 = 0.0;
	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			R2 += r(i,j)*r(i,j);
		}
	}
	double psi = exp(-0.5*alpha*R2)

	// jastrow factor
	for(int i1 = 0; i1 < N_-1; ++i1){
		for(int i2 = i1+1; i2 < N_; ++i2){
			R2 = 0.0;
			for(int j = 0; j < dim_; ++j){
				R2 += pow(r(r1,j)-r(i2,j),2.0);
			}
			psi *= exp(R2/(1.0+beta*R2));
		}
	}

	return psi;
}

double CFermiSystem::calc_local_energy(){

	double h = 1.0E-5
	double E_kinetic = 0.0, E_potential = 0.0;
	mat 
	mat r_plus = r_, r_minus = r_;


}

