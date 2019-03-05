#include "boson_system.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){

	int number_bosons = 10;
	int max_variation = atoi(argv[1]);
	double position_step = 1.0;
	double mass = 1.0;
	double hard_core_diameter = 0.0;
	double omega_xy = 1.0;
	double omega_z = 1.0;
	double alpha0 = 1.0, alphaf = 10.0;
	double beta0 = 0.1, betaf = 0.6;

	int number_MC_cycles = atoi(argv[2]);

	CBosonSystem Bosons(number_bosons, max_variation, position_step, mass, hard_core_diameter, omega_xy, omega_z, alpha0, alphaf, beta0, betaf);
	Bosons.montecarlo_sampling(number_MC_cycles,"bosons");

	return 0;
}