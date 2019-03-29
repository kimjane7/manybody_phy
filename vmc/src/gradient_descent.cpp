#include "gradient_descent.h"

CGradientDescent::CGradientDescent(vec &variational_params, mat &data, double (*function_to_minimize)(int number_MC_cycles)){

	params0_ = variational_params;
	data_ = data;
	func_ = function_to_minimize;

	N_ = data_.n_rows();                  // number of data points
	D_ = params0_.n_elem();               // number of variational parameters

	tolerance_ = 1E-10;

}


CGradientDescent::steepest(vec (*gradient_wrt_params)()){

	params_ = params0_;
	gradient_ = zeros<vec>(D_);

	func_diff_ = 

	do{
		func_diff_

		gradient_ = 
	}


}