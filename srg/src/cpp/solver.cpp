// calculates Ecorr for pairing model for various 
// interaction strengths g using different methods

#include "solver.h"


CSolver::CSolver(int particles, int sp_states, double d, double gmin, double gmax, string filename){

	particles_ = particles;
	sp_states_ = sp_states;
	d_ = d;
	gmin_ = gmin;
	gmax_ = gmax;
	gstep_ = (gmax-gmin)/100.0;
	filename_ = filename;
}


void CSolver::MBPT2(){

	ofstream outfile;
	outfile.open(filename_+"_mbpt2.dat");
	outfile << "# d = " << d_ << endl;
	outfile << "# g, Ecorr from MBPT2, analytical Ecorr (pg. 365)" << endl;

	for(double g = gmin_+0.5*gstep_; g < gmax_; g += gstep_){

		CFermionSystem system(particles_, sp_states_, d_, g);

		double Ecorr = 0.0;
		for(int a = system.F_; a < system.N_; ++a){
			for(int b = system.F_; b < system.N_; ++b){
				for(int i = 0; i < system.F_; ++i){
					for(int j = 0; j < system.F_; ++j){
						ivec holes = {i,j};
						ivec parts = {a,b};
						Ecorr += system.v(a,b,i,j)*system.v(a,b,i,j)/system.eps(holes,parts);
					}
				}
			}
		}
		Ecorr = 0.25*Ecorr;

		// analytical solution (pg. 365)
		double soln = -0.25*g*g*(2/(4+g)+1/(6+g)+1/(2+g));
		outfile << g << "\t" << Ecorr << "\t" << soln << endl;
	}

	outfile.close();
}

void CSolver::SRG(){

	ofstream outfile;
	outfile.open(filename_+"_srg.dat");
	outfile << "# d = " << d_ << endl;
	outfile << "# g, Ecorr using SRG, six eigenvalues of H" << endl;

		for(double g = gmin_+0.5*gstep_; g < gmax_; g += gstep_){

		CFermionSystem system(particles_, sp_states_, d_, g);
		
		// set up Hamiltonian
		mat H0 = zeros<mat>(6,6);
		for(int i = 0; i < 6; ++i){
			for(int j = 0; j < 6; ++j){
				if(i+j != 5) H0(i,j) = -0.5*g; 
			}
		}
		H0(0,0) = 2*d_-g;
		H0(1,1) = 4*d_-g;
		H0(2,2) = 6*d_-g;
		H0(3,3) = 6*d_-g;
		H0(4,4) = 8*d_-g;
		H0(5,5) = 10*d_-g;

		CSRG solver(H0, 6, 170, 10.0, 0.001);
		mat H = solver.srg();

		outfile << g << "\t" << H(0,0)-2.0+g << "\t";
		outfile << H(0,0) << "\t" << H(1,1) << "\t";
		outfile << H(2,2) << "\t" << H(3,3) << "\t";
		outfile << H(4,4) << "\t" << H(5,5) << endl;
	}

	outfile.close();

}