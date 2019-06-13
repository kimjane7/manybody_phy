#include "boson_system.h"
#include "RBM.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){



	int dimension = 2;
	int number_particles = 2;
	int number_hidden = 2;
	int number_MC_cycles = 1000000;
	double hard_core_diameter = 0.5;
	double tolerance = 1E-8;
	vec omega = ones<vec>(dimension);	
	
	
	RBM WaveFunction(number_particles, number_hidden, hard_core_diameter, omega);
	cout << "test 1" << endl;
	WaveFunction.steepest_gradient_descent(number_MC_cycles, tolerance, "test.dat");
	cout << "success" << endl;

	return 0;
}