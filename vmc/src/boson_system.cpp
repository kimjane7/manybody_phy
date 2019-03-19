#include "boson_system.h"

CBosonSystem::CBosonSystem(int dimension, int number_bosons, int number_epochs, int batch_size, double hard_core_diameter, vec omega){

	D_ = dimension;
	N_ = number_bosons;
	E_ = number_epochs;
	b_ = batch_size;
	B_ = (int) N_/b_;	               // number of batches for SGD
	a_ = hard_core_diameter;

	timestep_ = 0.05;			       // Fokker-Planck equation parameters
	diff_coeff_ = 0.5;
	psi_ = 0.0;					       // trial wave function
	psi_new_ = 0.0;

	alpha_.zeros(D_);                  // variational parameters
	r_.zeros(N_,D_);                   // positions
	r_new_.zeros(N_,D_);
	qforce_.zeros(N_,D_);	      	   // quantum force
	qforce_new_.zeros(N_,D_);

	batches_.zeros(B_,b_);             // indices of batches
	for(int k = 0; k < B_; ++k){
		for(int l = 0; l < b_; ++l){
			batches_(k,l) = k*b_+l;
		}
	}

	omega2_ = omega % omega;           // frequencies squared
	if(omega.n_elem != D_){
		cout << "WARNING: number of frequencies do not match dimension." << endl;
	}
}

// stochastic gradient descent
void CBosonSystem::stochastic_gradient_descent(){

	int j = 0, k;
	double t0 = 1.0, t1 = 10.0;		  // step length parameters
	double gammaj = t0/t1;            // step length
	vec gradient = zeros<vec>(D_);    // gradient of energy

	// loop over epochs
	for(int e = 0; e < E_; ++e){

		// loop over number of batches
		for(int b = 0; b < B_; ++b){

			// pick kth batch at random
			k = randint(B_);

			// compute gradient using the data in kth batch
			gradient = calc_derivative_local_energy(r_,alpha_)

			// compute new suggestion for alpha
			alpha_ -= gammaj*gradient;

			// update step length
			gammaj = t0/(e*M+m+t1);
			j += 1;
		}
	}
}

// with importance sampling
void CBosonSystem::energy_minimization(int number_MC_cycles, vec alpha, string filename){

	double E = 0.0, E2 = 0.0, deltaE = 0.0, E_err;
	vec dE = zeros<vec>(D_);
	vec deltapsi = zeros<vec>(D_);
	vec dpsi = zeros<vec>(D_);
	vec dpsiE = zeros<vec>(D_);

	// randomly position particles
	random_initial_positions();

	// calculate initial wave function and quantum force
	psi_ = calc_trial_wavefunction(r_,alpha);
	qforce_ = calc_quantum_force(r_,alpha);

	// loop over monte carlo cycles
	for(int m = 0; m < number_MC_cycles; ++m){

		for(int i = 0; i < N_; ++i){

			// move ith particle
			random_new_position(i);

			// calculate new wave function and quantum force
			psi_new_ = calc_trial_wavefunction(r_new_,alpha);
			qforce_new_ = calc_quantum_force(r_new_,alpha);

			// Metropolis-Hastings test
			if(rand01() <= acceptance_ratio(i)){
				r_.row(i) = r_new_.row(i);
				qforce_.row(i) = qforce_new_.row(i);
				psi_ = psi_new_;
			}
		}

		// calculate energy and derivatives
		deltaE = calc_local_energy(r_,alpha);
		dpsi = calc_derivative_wavefunction(r_,alpha);
		E += deltaE;
		E2 += deltaE*deltaE;
		deltapsi += dpsi;
		dpsiE += dpsi*deltaE;
	}

	// calculate mean, variance, error
	E /= number_MC_cycles;
	E2 /= number_MC_cycles;
	dpsiE /= number_MC_cycles;
	deltapsi /= number_MC_cycles;
	E_err = sqrt((E2-E*E)/number_MC_cycles);
	dE = 2.0*(dpsiE-deltapsi*E);
}


// calculate trial wavefunction
double CBosonSystem::calc_trial_wavefunction(mat r, vec alpha){

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

		ri2 = r.row(i) % r.row(i);
		sum += norm(dot(alpha,ri2));
	}
	psi *= exp(-sum);

	return psi;
}

// calculate local energy (see equations pdf in doc folder)
double CBosonSystem::calc_local_energy(mat r, vec alpha){

	double EL = 0.0, prefactor, rij;
	vec alpha2 = alpha%alpha;
	vec one = ones<vec>(D_);
	vec sum = zeros<vec>(D_);
	vec Ri = zeros<vec>(D_);
	vec Rj = zeros<vec>(D_);


	EL += N_*dot(alpha,one);

	for(int i = 0; i < N_; ++i){

		Ri = r.row(i); // check this is what i expect

		EL += 0.5*dot(omega2_, Ri%Ri);  // harmonic oscillator part
		EL -= 2.0*dot(alpha2, Ri%Ri);

		for(int j = 0; j < N_; ++j){
			if(i != j){

				Rj = r.row(j);
				rij = distance(r,i,j);
				prefactor = a_/(rij*rij*(rij-a_));

				sum += prefactor*(Ri-Rj);				
			}

		}

		EL -= 0.5*dot(sum,sum);
	}

	for(int i = 0; i < N_-1; ++i){
		for(int j = i+1; j < N_; ++j){

			Rj = r.row(j);
			rij = distance(r,i,j);

			prefactor = a_/(rij*rij*(rij-a_));

			EL += prefactor*(((3.0-D_)*rij+(D_-2.0)*a)/(rij-a_)+4.0*dot(alpha%Ri,Ri-Rj));
		}
	}

	return EL;
}

// calculate quantum force for each particle
mat CBosonSystem::calc_quantum_force(mat r, vec alpha){

	double prefactor, rij;
	vec Ri = zeros<vec>(D_);
	vec Rj = zeros<vec>(D_);
	mat qforce = zeros<mat>(N_,D_);

	for(int i = 0; i < N_; ++i){

		Ri = r.row(i);

		qforce.row(i) = -4.0*(alpha%Ri);

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

// calculate derivative of trial wavefunction wrt variational parameters
vec CBosonSystem::calc_derivative_wavefunction(mat r, vec alpha){

	double sum;
	vec dpsi = zeros<vec>(D_);

	for(int d = 0; d < D_; ++d){

		sum = 0.0;
		for(int i = 0; i < N_; ++i){
			sum += r(i,d)*r(i,d);
		}
		dpsi(d) = -sum;
	}
	dpsi = calc_trial_wavefunction(r,alpha)*dpsi;

	return dpsi;
}

// calculate derivative of local energy wrt variational parameters for batch only
vec CBosonSystem::calc_derivative_local_energy(int batch_index, mat r, vec alpha){

	int k = batch_index;
	double sum, rij;	
	vec dEL = zeros<vec>(D_);

	for(int d = 0; d < D_; ++d){

		sum = b_;

		// loop through particles in batch
		for(int l = 0; l < b_; ++l){

			i = batches_(k,l);
			sum -= 4.0*alpha(d)*r(i,d)*r(i,d);

			for(int j = 0; j < N_; ++j){
				if(i != j){

					rij = distance(r,i,j);
					sum += 2.0*a_*r(i,d)*(r(i,d)-r(j,d))/(rij*rij*(rij-a_));
				}
			}
		}

		dEL(d) = sum;
	}

	return dEL;
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
		greens += 0.5*(r_(i,d)-r_new_(i,d)(qforce_new_(i,d)+qforce_(i,d)));
		greens += 0.25*diff_coeff_*timestep_*(pow(qforce_(i,d),2.0)-pow(qforce_new_(i,d),2.0));
	}
	greens = exp(greens);

	return greens*psi_new_*psi_new_/(psi_*psi_);
}

// distance between ith and jth bosons
double CBosonSystem::distance(mat r, int i, int j){

	vec rij = r.row(i)-r.row(j);

	return norm(rij,2);
}

// uniform real rng
double CBosonSystem::rand01(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}

// uniform int rng
double CBosonSystem::randint(int max){

	uniform_int_distribution<int> dist(0,max-1);
	return dist(rng_);
}

// normal rng
double CBosonSystem::randnorm(){

	normal_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}