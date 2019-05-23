#include "RBM.h"

CRBM::CRBM(int number_hidden, vec input){

	M_ = input.n_elem;
	N_ = number_hidden;
	x_ = input;

	// set random initial hidden layer, biases, and weights
	arma::arma_rng::set_seed_random();
	h_ = randi<ivec>(N_, distr_param(0,1));
	a_.randn(M_);
	b_.randn(N_);
	W_.randn(M_,N_);

}

double CRBM::energy(){

	return 0.5*norm(x_-a_)-dot(b_,h_)-dot(x_,W_*h_);
}