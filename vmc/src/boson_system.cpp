#include "boson_system.h"

CBosonSystem::CBosonSystem(int dimension, int number_bosons, double hard_core_diameter, vec omega){

	// ints
	D_ = dimension;
	N_ = number_bosons;
	
	// doubles
	a_ = hard_core_diameter;
	E_ = 0.0;						   // mean energy
	E_err_ = 0.0;					   // standard deviation of energy
	timestep_ = 0.05;			       // Fokker-Planck equation parameters
	diff_coeff_ = 0.5;
	psi_ = 0.0;					       // trial wave function
	psi_new_ = 0.0;

	// vecs
	delta_E_.zeros(D_); 			   // mean gradient of energy
	omega2_ = omega % omega;           // frequencies squared
	alpha_.zeros(D_);                  // variational parameters

	// mats
	r_.zeros(N_,D_);                   // positions
	r_new_.zeros(N_,D_);
	qforce_.zeros(N_,D_);	      	   // quantum force
	qforce_new_.zeros(N_,D_);

}

// steepest gradient descent
void CBosonSystem::steepest_gradient_descent(int number_MC_cycles, double tolerance, vec alpha0, string filename){

	int k = 0; 						   // number of iterations for convergence
	double gammak = 1.0;               // step length
	double E_last;					   // place holder for gs energy of last iteration
	alpha_ = alpha0;				   // initial variational parameters
	E_ = 0.0;

	// open file
	ofstream outfile;
	outfile.open(filename);
	outfile << "# number of MC cycles = " << number_MC_cycles << endl;
	outfile << "# tolerance = " << tolerance << endl;
	outfile << "# E, E_error, gradient of E, variational parameters" << endl;

	// descend until difference in energy is less than tolerance
	do{

		// store value of ground state energy of last iteration
		E_last = E_;

		// calculate ground state energy and stats
		variational_energy(number_MC_cycles);

		// print stats and variational params
		outfile << left << setw(15) << setprecision(5) << E_;
		outfile << left << setw(15) << setprecision(5) << E_err_;

		for(int d = 0; d < D_; ++d){
			outfile << left << setw(15) << setprecision(5) << delta_E_(d);
		}
		for(int d = 0; d < D_; ++d){
			outfile << left << setw(15) << setprecision(5) << alpha_(d);
		}
		outfile << endl;

		// use delta_E_ as approx. to gradient of local energy
		// compute new suggestion for variational parameters
		alpha_ -= gammak*delta_E_;
		
		// count iterations
		k += 1;

	}while(abs(E_-E_last) > tolerance);

	cout << "iterations: " << k << endl;

	outfile.close();
}


// importance sampling
void CBosonSystem::variational_energy(int number_MC_cycles){

	E_ = 0.0;
	double E2 = 0.0, EL;
	vec gradient_psi = zeros<vec>(D_);
	vec delta_psi = zeros<vec>(D_);
	vec delta_psiE = zeros<vec>(D_);
	
	// randomly position particles
	random_initial_positions();

	// calculate initial wave function and quantum force
	psi_ = calc_trial_wavefunction(r_);
	qforce_ = calc_quantum_force(r_);


	// loop over monte carlo cycles
	for(int m = 0; m < number_MC_cycles; ++m){

		for(int i = 0; i < N_; ++i){

			// move ith particle
			random_new_position(i);

			// calculate new wave function and quantum force
			psi_new_ = calc_trial_wavefunction(r_new_);
			qforce_new_ = calc_quantum_force(r_new_);

			// Metropolis-Hastings test
			if(rand01() <= acceptance_ratio(i)){

				r_.row(i) = r_new_.row(i);
				qforce_.row(i) = qforce_new_.row(i);
				psi_ = psi_new_;
			}
		}

		// calculate energy and gradient of wave function
		EL = calc_local_energy();
		E_ += EL;
		E2 += EL*EL;
		gradient_psi = calc_gradient_wavefunction();
		delta_psi += gradient_psi;
		delta_psiE += gradient_psi*EL;
	}

	// calculate mean and standard dev
	E_ /= number_MC_cycles;
	E2 /= number_MC_cycles;
	E_err_ = sqrt((E2-E_*E_)/number_MC_cycles);
	delta_psi /= number_MC_cycles;
	delta_psiE /= number_MC_cycles;

	// check gradient E goes to zero?
	delta_E_ = 2.0*(delta_psiE-delta_psi*E_);
}


// calculate trial wavefunction
double CBosonSystem::calc_trial_wavefunction(mat r){

	double psi = 1.0, sum = 0.0, rij;
	vec ri2;

	// no correlation part for a=0
	if(a_ > 0.0){
		for(int i = 0; i < N_; ++i){
			for(int j = 0; j < N_; ++j){
				if(i != j){

					rij = distance(r,i,j);

					if(rij <= a_) return 0.0;
					else psi *= 1.0-a_/rij;
				}
			}
		}
	}

	// elliptical harmonic oscillator part
	for(int i = 0; i < N_; ++i){

		ri2 = vectorise(r.row(i) % r.row(i));
		sum += dot(alpha_,ri2);

	}
	psi *= exp(-sum);	

	return psi;
}

// calculate local energy
double CBosonSystem::calc_local_energy(){

	double EL = 0.0, prefactor, rij;
	vec alpha2 = alpha_%alpha_;
	vec one = ones<vec>(D_);
	vec sum = zeros<vec>(D_);
	vec Ri = zeros<vec>(D_);
	vec Rj = zeros<vec>(D_);


	EL += N_*dot(alpha_,one);

	for(int i = 0; i < N_; ++i){

		Ri = vectorise(r_.row(i));

		EL += 0.5*dot(omega2_, Ri%Ri);
		EL -= 2.0*dot(alpha2, Ri%Ri);

		for(int j = 0; j < N_; ++j){
			if(i != j){

				Rj = vectorise(r_.row(j));
				rij = distance(r_,i,j);
				prefactor = a_/(rij*rij*(rij-a_));

				sum += prefactor*(Ri-Rj);				
			}

		}

		EL -= 0.5*dot(sum,sum);
	}

	for(int i = 0; i < N_-1; ++i){

		Ri = vectorise(r_.row(i));

		for(int j = i+1; j < N_; ++j){

			Rj = vectorise(r_.row(j));
			rij = distance(r_,i,j);

			prefactor = a_/(rij*rij*(rij-a_));

			EL += prefactor*(((3.0-D_)*rij+(D_-2.0)*a_)/(rij-a_)+4.0*dot(alpha_%Ri,Ri-Rj));
		}
	}

	return EL;
}

// calculate quantum force for each particle
mat CBosonSystem::calc_quantum_force(mat r){

	double prefactor, rij;
	rowvec Ri = zeros<rowvec>(D_);
	rowvec Rj = zeros<rowvec>(D_);
	mat qforce = zeros<mat>(N_,D_);

	for(int i = 0; i < N_; ++i){

		Ri = r.row(i);
		qforce.row(i) = -4.0*(conv_to<rowvec>::from(alpha_)%Ri);

		for(int j = 0; j < N_; ++j){
			if(i != j){

				Rj = r.row(j);
				rij = distance(r,i,j);
				prefactor = a_/(rij*rij*(rij-a_));

				qforce.row(i) += 2.0*prefactor*(Ri-Rj);				
			}
		}

	}

	return qforce;
}

// calculate gradient of trial wavefunction wrt variational parameters
vec CBosonSystem::calc_gradient_wavefunction(){

	double sum;
	vec gradient_psi = zeros<vec>(D_);

	for(int d = 0; d < D_; ++d){

		sum = 0.0;
		for(int i = 0; i < N_; ++i){
			sum += r_(i,d)*r_(i,d);
		}
		gradient_psi(d) = -sum;
	}
	gradient_psi = calc_trial_wavefunction(r_)*gradient_psi;

	return gradient_psi;
}

// calculate gradient of local energy wrt variational parameters
vec CBosonSystem::calc_gradient_local_energy(){

	double sum, rij;	
	vec gradient_EL = zeros<vec>(D_);

	// loop through components of gradient
	for(int d = 0; d < D_; ++d){

		sum = N_;

		for(int i = 0; i < N_-1; ++i){

			sum -= 4.0*alpha_(d)*r_(i,d)*r_(i,d);

			for(int j = i+1; j < N_; ++j){

				rij = distance(r_,i,j);
				sum += 4.0*a_*r_(i,d)*(r_(i,d)-r_(j,d))/(rij*rij*(rij-a_));
			}
		}

		gradient_EL(d) = sum;
	}

	return gradient_EL;
}

// set random initial positions
void CBosonSystem::random_initial_positions(){

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < D_; ++j){
			r_(i,j) = sqrt(timestep_)*randnorm();
		}
	}
}

// get new position of ith particle
void CBosonSystem::random_new_position(int i){

	for(int d = 0; d < D_; ++d){
		r_new_(i,d) = r_(i,d)+diff_coeff_*qforce_(i,d)*timestep_+randnorm()*sqrt(timestep_);
	}
}

// acceptance ratio for Metropolis-Hastings algorithm
double CBosonSystem::acceptance_ratio(int i){

	double greens = 0.0;

	for(int d = 0; d < D_; ++d){
		greens += 0.5*(r_(i,d)-r_new_(i,d)*(qforce_new_(i,d)+qforce_(i,d)));
		greens += 0.25*diff_coeff_*timestep_*(qforce_(i,d)*qforce_(i,d)-qforce_new_(i,d)*qforce_new_(i,d));
	}
	greens = exp(greens);

	return greens*psi_new_*psi_new_/(psi_*psi_);
}

// distance between ith and jth bosons
double CBosonSystem::distance(mat r, int i, int j){

	vec rij = vectorise(r.row(i)-r.row(j));

	return norm(rij,2);
}

// uniform int rng
int CBosonSystem::randint(int max){

	uniform_int_distribution<int> dist(0,max-1);
	return dist(rng_);
}

// uniform real rng
double CBosonSystem::rand01(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}

// normal rng
double CBosonSystem::randnorm(){

	normal_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}
