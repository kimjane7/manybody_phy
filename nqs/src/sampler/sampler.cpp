#include "sampler.h"

Sampler::Sampler(int seed, int n_cycles, int n_samples, 
	NeuralQuantumState &NQS, Hamiltonian &H, Optimizer &O, 
	string filename, string block_filename):
	NQS_(NQS), H_(H), O_(O){

    random_engine_ = mt19937_64(seed);
    n_cycles_ = n_cycles;
    n_samples_ = n_samples;
    outfile_.open(filename);
    block_outfile_.open(block_filename);
}

void Sampler::optimize(){

	bool accepted;
	int n_accepted, n_samples_effective;
	double ratio_accepted;
	double EL, EL_mean, EL2_mean, EL_var;
	VectorXd grad_logpsi(O_.n_params_);
	VectorXd grad_logpsi_mean(O_.n_params_);
	VectorXd EL_grad_logpsi(O_.n_params_);
	VectorXd EL_grad_logpsi_mean(O_.n_params_);
	VectorXd grad_EL(O_.n_params_);
	
	// optimization iterations
	for(int cycles = 0; cycles < n_cycles_; ++cycles){

		n_accepted = 0;
		n_samples_effective = 0;
		EL_mean = 0.0;
		EL2_mean = 0.0;
		grad_logpsi_mean.setZero();
		EL_grad_logpsi_mean.setZero();

		// samples for estimating gradient
		for(int samples = 0; samples < n_samples_; ++samples){

			sample(accepted);

			// skip some samples
			if(samples > 0.1*n_samples_){

				EL = H_.calc_local_energy();
				grad_logpsi = H_.calc_gradient_logpsi();

				// add up values for expectation values
				EL_mean += EL;
				EL2_mean += EL*EL;
				grad_logpsi_mean += grad_logpsi;
				EL_grad_logpsi_mean += EL*grad_logpsi;

				if(accepted) n_accepted++;
				n_samples_effective++;

			}
		}

		// calculate expectation values
		EL_mean /= n_samples_effective;
		EL2_mean /= n_samples_effective;
		grad_logpsi_mean /= n_samples_effective;
		EL_grad_logpsi_mean /= n_samples_effective;

		// calculate variance and ratio accepted
		EL_var = EL2_mean-EL_mean*EL_mean;
		ratio_accepted = n_accepted/(double)n_samples_effective;

		// calculate gradient
		grad_EL = 2.0*(EL_grad_logpsi_mean-EL_mean*grad_logpsi_mean);
		cout << cycles << "\t" << grad_EL << endl;

		// update weights
		O_.optimize_weights(grad_EL, NQS_);

		outfile_ << cycles << "\t" << EL_mean << "\t" << EL_var << "\t" << ratio_accepted << "\n";
	}

	outfile_.close();
}