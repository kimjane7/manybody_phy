#include "boson_system.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){

	int dimension = 1;
	int number_bosons = 2;
	double hard_core_diameter = 0.0;
	vec omega = {1.0};

	CBosonSystem Bosons(dimension, number_bosons, hard_core_diameter, omega);
	
	int number_MC_cycles = 10;
	double tolerance = 1E-8;
	vec alpha0 = {1.0};

	Bosons.steepest_gradient_descent(number_MC_cycles, tolerance, alpha0, "bosons_N2_D1.dat");

	return 0;
}