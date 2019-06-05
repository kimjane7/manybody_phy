#include "boson_system.h"
#include "RBM.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){



	int dimension = 2;
	int number_particles = 5;
	/*
	int number_bosons = 2;
	double hard_core_diameter = 0.0;
	vec omega = ones<vec>(dimension);

	CBosonSystem Bosons(dimension, number_bosons, hard_core_diameter, omega);
	
	int number_MC_cycles = 10;
	double tolerance = 1E-8;
	vec alpha0 = ones<vec>(dimension);

	Bosons.steepest_gradient_descent(number_MC_cycles, tolerance, alpha0, "bosons_N2_D2.dat");
	*/

	int number_hidden = 3;
	vec omega = ones<vec>(dimension);
	RBM WaveFunction(number_particles, number_hidden, omega);
	cout << WaveFunction.calc_local_energy() << endl;


	return 0;
}