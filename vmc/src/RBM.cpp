#include "RBM.h"

RBM::RBM(int number_particles, int number_hidden, double hard_core_diameter, vec omega){

	P_ = number_particles;
	D_ = omega.n_elem;
	M_ = P_*D_;
	N_ = number_hidden;
	a0_ = hard_core_diameter;

	Omega2_.zeros(M_);
	for(int i = 0; i < M_; ++i){
		Omega2_(i) = pow(omega(i%D_),2.0);
	}

	arma::arma_rng::set_seed_random();

	E_ = 0.0;                          // mean energy
	EL_ = 0.0;                         // local energy
	E_err_ = 0.0;                      // standard deviation of energy
	delta_E_.zeros(M_+N_+M_*N_);       // mean gradient of energy wrt variational params
	timestep_ = 0.05;			       // Fokker-Planck equation time step
	diff_coeff_ = 0.5;                 // Fokker-Planck equation diffusion coefficient

	psi_ = 0.0;					       // trial wave function
	psi_new_ = 0.0;
	qforce_.zeros(M_);                 // quantum force
	qforce_new_.zeros(M_);             

	// precalculated quantities - see documentation (rbm.pdf)
	B_.zeros(N_);                      // Bj's stored in vector   
	f_.zeros(N_);                      // fj's stored in vector
}

// steepest gradient descent
void RBM::steepest_gradient_descent(int number_MC_cycles, double tolerance, string filename){

	int k = 0; 						   // number of iterations for convergence
	double gammak = 1.0;               // learning rate
	double E_last;					   // place holder for gs energy of last iteration
	E_ = 0.0;

	// open file
	ofstream outfile;
	outfile.open(filename);
	outfile << "# number of MC cycles = " << number_MC_cycles << endl;
	outfile << "# tolerance = " << tolerance << endl;

	// initial variational parameters
	set_initial_params();

	// descend until difference in energy is less than tolerance
	do{

		// store value of ground state energy of last iteration
		E_last = E_;

		// calculate ground state energy and stats
		variational_energy(number_MC_cycles);

		// print stats
		outfile << left << setw(15) << setprecision(5) << E_;
		outfile << left << setw(15) << setprecision(5) << E_err_;
		outfile << left << setw(15) << setprecision(5) << norm(delta_E_) << endl;

		// update variational parameters
		theta_ -= gammak*delta_E_;
		separate_params();
		
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
	vec gradient_psi = zeros<vec>(M_+N_+M_*N_);
	vec delta_psi = zeros<vec>(M_+N_+M_*N_);
	vec delta_psiE = zeros<vec>(M_+N_+M_*N_);
	
	// initial visible and hidden nodes
	set_initial_nodes();

	// calculate initial wave function and quantum force
	psi_ = calc_trial_wavefunction(x_);
	qforce_ = calc_quantum_force(x_);

	// loop over monte carlo cycles
	for(int m = 0; m < number_MC_cycles; ++m){

		for(int k = 0; k < P_; ++k){

			// move kth particle
			random_new_position(k);

			// calculate new wave function and quantum force
			psi_new_ = calc_trial_wavefunction(x_new_);
			qforce_new_ = calc_quantum_force(x_new_);

			// Metropolis-Hastings test
			if(rand01() <= acceptance_ratio(k)){

				x_ = x_new_;
				qforce_ = qforce_new_;
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

	// gradient of energy wrt variational parameters
	delta_E_ = 2.0*(delta_psiE-delta_psi*E_);
}

// calculate local energy
double RBM::calc_local_energy(){

	int k, d;
	double foo, denom, Rpk;
	double EL = M_ + dot(Omega2_,x_%x_);

	// precalculate f
	store_factors(x_);
	
	for(int j = 0; j < N_; ++j){

		EL -= exp(-B_(j))*pow(f_(j)*norm(W_.col(j)),2.0);
	}

	// calculate the rest of EL
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

// calculate trial wavefunction
double RBM::calc_trial_wavefunction(mat x){

	double Rpq, sum = 0.0;

	// precalculate B
	store_factors(x);

	// interaction part
	if(a0_ > 0.0){
		for(int p = 0; p < P_; ++p){
			for(int q = 0; q < p; ++q){

				Rpq = distance(x,p,q);

				if(Rpq <= a0_) return 0.0;
				else sum += log(1.0-a0_/Rpq);
			}
		}
	}

	// RBM part
	sum -= 0.5*pow(norm(x-a_),2.0);
	for(int j = 0; j < N_; ++j){
		sum += log(1.0+exp(B_(j)));
	}

	return exp(sum);
}

// calculate quantum force for each particle
vec RBM::calc_quantum_force(mat x){

	int k;
	vec qforce = zeros<vec>(M_);

	for(int i = 0; i < M_; ++i){

		k = floor(i/D_);

		qforce(i) = a_(i)-x_(i)+dot(vectorise(W_.row(i)),f_);

		if(a0_ > 0.0){
			
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

// get new position of kth particle
void RBM::random_new_position(int k){

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
double RBM::r(mat x, int k, int p){

	double distance = 0.0;

	for(int d = 0; d < D_; ++d){
		distance += pow(x(D_*k+d)-x(D_*p+d),2.0);
	}

	return sqrt(distance);
}

// store B and f for given x
void RBM::store_factors(vec x){

	for(int j = 0; j < N_-1; ++j){

		B_(j) = b_(j) + dot(x,W_.col(j));
		f_(j) = 1.0/(exp(-B_(j))+1.0);
	}
}

// store all variational parameters a, b, W in one vector theta
void RBM::combine_params(){

	int index;

	theta_.head(M_) = a_;
	theta_.subvec(M_,M_+N_-1) = b_;

	// store weights column-wise
	for(int j = 0; j < N_; ++j){
		index = M_+N_+j*M_;
		theta_.subvec(index,index+M_-1) = W_.col(j);
	}
}

// get variational parameters a, b, W from one vector theta
void RBM::separate_params(){

	int index;

	a_ = theta_.head(M_);
	b_ = theta.subvec(M_,M_+N_-1);

	// separate weights column-wise
	for(int j = 0; j < N_; ++j){
		index = M_+N_+j*M_;
		W_.col(j) = theta_.subvec(index,index+M_-1);
	}
}

void RBM::set_initial_nodes(){

	x_.randn(M_);
	x_new_.zeros(M_);
	h_ = randi<ivec>(N_, distr_param(0,1));
}

// set random initial positions, hidden layer, biases, and weights
void RBM::set_initial_params(){

	// biases and weights
	a_.randn(M_);
	b_.randn(N_);
	W_.randn(M_,N_);

	// vectorized parameters
	theta_ = zeros<vec>(M_+N_+M_*N_);
	combine_params();
}