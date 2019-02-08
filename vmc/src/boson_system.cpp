#include "boson_system.h"

CBosonSystem::CBosonSystem(int dimension, int number_bosons, int max_variation, double position_step, 
	                       double alpha0, double alphaf, double beta0, double betaf){

	dim_ = dimension;
	N_ = number_bosons;
	max_ = max_variation;
	step_ = position_step;
	psi_ = 0.0;
	psi_new_= 0.0;

	r_.zeros(N_,dim_);
	r_new_.zeros(N_,dim_);
	alpha_ = linspace(alpha0,alphaf,max_);
	beta_ = linspace(beta0,betaf,max_);
	E_.zeros(max_,max_);
	E_err_.zeros(max_,max_);
}

void CBosonSystem::montecarlo_sampling(int number_MC_cycles, string filename){

	// open file
	ofstream outfile;
	outfile.open(filename+"_"+to_string(max_)+"_"+to_string(number_MC_cycles)+".dat");
	outfile << "# alpha, beta, E, E_err" << endl;

	// heading
	//printf("%10s %10s %10s %10s \n\n","alpha","beta","energy","error");

	// loop over various parameter values
	for(int a = 0; a < max_; ++a){
		for(int b = 0; b < max_; ++b){

			double E = 0.0, E2 = 0.0, DeltaE = 0.0, E_err;

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
				if(rand01_() < pow(psi_new_/psi_,2.0)){
					r_ = r_new_;
					psi_ = psi_new_;
					DeltaE = calc_local_energy(alpha_(a),beta_(b));
				}
				E += DeltaE;
				E2 += DeltaE*DeltaE;
			}

			// calculate mean, variance, error
			E = E/number_MC_cycles;
			E2 = E2/number_MC_cycles;
			E_err = sqrt((E2-E*E)/number_MC_cycles);

			// store results
			E_(a,b) = E;
			E_err_(a,b) = E_err;

			// print results
			//printf("%10.3lf %10.3lf %10.3lf %10.3lf \n",alpha_(a),beta_(b),E_(a,b),E_err_(a,b));

			outfile << left << setw(10) << setprecision(5) << alpha_(a);
			outfile << left << setw(10) << setprecision(5) << beta_(a);
			outfile << left << setw(10) << setprecision(5) << E_(a,b);
			outfile << left << setw(10) << setprecision(5) << E_err_(a,b) << endl;		}
	}

	outfile.close();
}

void CBosonSystem::random_initial_positions(){


	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			r_(i,j) = step_*(rand01_()-0.5);
		}
	}
}

void CBosonSystem::random_trial_positions(){

	r_new_ = zeros<mat>(max_,max_);

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			r_new_(i,j) = r_(i,j)+step_*(rand01_()-0.5);
		}
	}
}

// two (more?) electrons in harmonic oscillator potential
double CBosonSystem::calc_trial_wavefunction(mat r, double alpha, double beta){

	// slater determinant
	double R2 = 0.0;
	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){
			R2 += r(i,j)*r(i,j);
		}
	}
	double psi = exp(-0.5*alpha*R2);

	// jastrow factor
	for(int i1 = 0; i1 < N_-1; ++i1){
		for(int i2 = i1+1; i2 < N_; ++i2){
			R2 = 0.0;
			for(int j = 0; j < dim_; ++j){
				R2 += pow(r(i1,j)-r(i2,j),2.0);
			}
			psi *= exp(R2/(1.0+beta*R2));
		}
	}

	return psi;
}

double CBosonSystem::calc_local_energy(double alpha, double beta){

	double omega2 = 1.0; // harmonic oscillator angular frequency squared
	double h = 0.001, h2 = h*h;
	double EL_kinetic = 0.0, EL_potential = 0.0, R2 = 0.0;
	double psi_plus, psi_minus;
	mat r_plus = r_, r_minus = r_;

	// calculate local kinetic energy
	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < dim_; ++j){

			// take small steps in jth dimension about the position of particle i 
			r_plus(i,j) += h;
			r_minus(i,j) -= h;
			
			// approximate second derivative in kinetic term
			psi_plus = calc_trial_wavefunction(r_plus,alpha,beta);
			psi_minus = calc_trial_wavefunction(r_minus,alpha,beta);
			EL_kinetic += psi_plus+psi_minus-2.0*psi_;

			// remove the steps
			r_plus(i,j) = r_(i,j);
			r_minus(i,j) = r_(i,j);
		}
	}
	EL_kinetic = 0.5*EL_kinetic/(h2*psi_);

	// calculate local potential energy
	// harmonic oscillator contribution
	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < N_; ++j){
			R2 += r_(i,j)*r_(i,j);
		}
	}
	EL_potential += 0.5*omega2*R2;

	// coulomb contribution (charge?)
	for(int i1 = 0; i1 < N_-1; ++i1){
		for(int i2 = i1+1; i2 < N_; ++i2){
			R2 = 0.0;
			for(int j = 0; j < dim_; ++j){
				R2 += pow(r_(i1,j)-r_(i2,j),2.0);
			}
			EL_potential += 1/sqrt(R2);
		}
	}
	EL_potential = EL_potential/psi_;

	return EL_kinetic+EL_potential;
}

double CBosonSystem::rand01_(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}