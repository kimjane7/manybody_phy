#include "RBM.h"

RBM::RBM(int number_particles, int number_hidden, vec omega){

	P_ = number_particles;
	D_ = omega.n_elem;
	M_ = P_*D_;
	N_ = number_hidden;


	// copied from boson_system
	a_ = hard_core_diameter;
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

	// store factors and frequencies in vectors for faster computation 
	// (see documentation - rbm.pdf)
	f_.zeros(N_);
	Omega2_.zeros(M_);
	for(int i = 0; i < M_; ++i){
		Omega2_(i) = pow(omega(i%D_),2.0);
	}

}

double RBM::calc_energy(){

	return 0.5*norm(x_-a_)-dot(b_,h_)-dot(x_,W_*h_);
}

double RBM::calc_local_energy(){

	vec Vj = zeros<vec>(M_);

	double EL = 0.5*(M_ - pow(norm(a_-x_),2.0) + dot(Omega2_,(x_%x_)));
	store_factors();

	for(int j = 0; j < N_; ++j){

		Vj = 0.5*W_.col(j) + (a_-x_);

		for(int jj = 0; jj < j; ++jj){
			Vj += f_(jj)*W_.col(jj);
		}

		EL -= f_(j)*dot(W_.col(j),Vj);
	}

	return EL;
}

double RBM::calc_wavefunction()

void RBM::store_factors(){

	double Bj;

	for(int j = 0; j < N_; ++j){

		Bj = b_(j) + dot(x_,W_.col(j));
		f_(j) = 1.0/(exp(-Bj)+1.0);
	}
}