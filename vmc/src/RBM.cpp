#include "RBM.h"

RBM::RBM(int number_particles, int number_hidden, vec omega){

	P_ = number_particles;
	D_ = omega.n_elem;
	M_ = P_*D_;
	N_ = number_hidden;


	// copied from boson_system
	a0_ = hard_core_diameter;
	E_ = 0.0;						   // mean energy
	E_err_ = 0.0;					   // standard deviation of energy
	timestep_ = 0.05;			       // Fokker-Planck equation parameters
	diff_coeff_ = 0.5;
	psi_ = 0.0;					       // trial wave function
	psi_new_ = 0.0;
	delta_E_.zeros(D_); 			   // mean gradient of energy


	// set random initial positions, hidden layer, biases, and weights
	arma::arma_rng::set_seed_random();
	x_.randn(M_);
	x_new_.zeros(M_);
	h_ = randi<ivec>(N_, distr_param(0,1));
	a_.randn(M_);
	b_.randn(N_);
	W_.randn(M_,N_);

	// store B, f, and frequencies in vectors for faster computation 
	// (see documentation - rbm.pdf)
	B_.zeros(N_);
	f_.zeros(N_);
	store_factors();
	Omega2_.zeros(M_);
	for(int i = 0; i < M_; ++i){
		Omega2_(i) = pow(omega(i%D_),2.0);
	}

}

// steepest gradient descent
void RBM::steepest_gradient_descent(int number_MC_cycles, double tolerance, string filename){

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
void RBM::variational_energy(int number_MC_cycles){

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
double RBM::calc_trial_wavefunction(mat r){

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
double RBM::calc_local_energy(){

	int k, d;
	double foo, denom, Rpi;
	double EL = M_ + dot(Omega2_,x_%x_);
	
	store_factors();

	for(int j = 0; j < N_; ++j){

		EL -= exp(-B_(j))*pow(f_(j)*norm(W_.col(j)),2.0);
	}

	for(int i = 0; i < M_; ++i){

		foo = a_(i)-x_(i)+dot(vectorise(W_.row(i)),f_)

		// hard-core interaction
		if(a0_ > 0.0){

			k = floor(i/D_);
			d = i%D_;

			for(int p = 0; p < P_; ++p){

				Rpi = distance(x_,i,p);

				if(p != k){

					denom = Rpi*Rpi*(Rpi-a0_);
					
					foo += a0_*(x_(i)-x_(D_*p+d))/denom;
					EL -= (a0_/denom)*(pow((x_(i)-x_(D_*p+d)),2.0)*(2.0*a0_-3.0*Rpi)+1.0);

				}
			}

			EL -= foo*foo;
		}
	}

	return 0.5*EL;
}

// calculate quantum force for each particle
mat RBM::calc_quantum_force(mat r){

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
vec RBM::calc_gradient_wavefunction(){

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
vec RBM::calc_gradient_local_energy(){

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
void RBM::random_initial_positions(){

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < D_; ++j){
			r_(i,j) = sqrt(timestep_)*randnorm();
		}
	}
}

// get new position of ith particle
void RBM::random_new_position(int i){

	for(int d = 0; d < D_; ++d){
		r_new_(i,d) = r_(i,d)+diff_coeff_*qforce_(i,d)*timestep_+randnorm()*sqrt(timestep_);
	}
}

// acceptance ratio for Metropolis-Hastings algorithm
double RBM::acceptance_ratio(int i){

	double greens = 0.0;

	for(int d = 0; d < D_; ++d){
		greens += 0.5*(r_(i,d)-r_new_(i,d)*(qforce_new_(i,d)+qforce_(i,d)));
		greens += 0.25*diff_coeff_*timestep_*(qforce_(i,d)*qforce_(i,d)-qforce_new_(i,d)*qforce_new_(i,d));
	}
	greens = exp(greens);

	return greens*psi_new_*psi_new_/(psi_*psi_);
}

// distance between the kth and pth bosons
double RBM::distance(mat x, int i, int p){

	int k = floor(i/D_);
	double distance = 0.0;

	for(int d = 0; d < D_; ++d){
		distance += pow(x(D_*k+d)-x(D_*p+d),2.0);
	}

	return sqrt(distance);
}

// store B and f
void RBM::store_factors(){

	for(int j = 0; j < N_-1; ++j){

		B_(j) = b_(j) + dot(x_,W_.col(j));
		f_(j) = 1.0/(exp(-B_(j))+1.0);
	}

}

/*
// uniform int rng
int RBM::randint(int max){

	uniform_int_distribution<int> dist(0,max-1);
	return dist(rng_);
}

// uniform real rng
double RBM::rand01(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}

// normal rng
double RBM::randnorm(){

	normal_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}
*/
