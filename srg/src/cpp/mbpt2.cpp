#include "system.h"
#include "solver.h"

int main(int argc, char *argv[]){

	CSolver PairingModel(4, 4, 1.0, -1.0, 1.0, "Ecorr_pairing");
	PairingModel.MBPT2();
	PairingModel.SRG();

	return 0;
}