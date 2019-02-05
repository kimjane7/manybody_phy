#include "fermi_system.h"

int main(int argc, char *argv[]){

	int max_variation = atoi(argv[1]);
	int number_MC_cycles = atoi(argv[2]);

	CFermiSystem QuantumDots(2,2,max_variation,1.0,1.0,10.0,0.1,0.6);
	QuantumDots.montecarlo_sampling(number_MC_cycles,"quantumdots");

	return 0;
}