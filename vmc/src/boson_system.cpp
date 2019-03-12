#include "boson_system.h"

CBosonSystem::CBosonSystem(int dimension, int number_bosons, double hard_core_diameter, vec omega){

	D_ = dimension;
	N_ = number_bosons;
	a_ = hard_core_diameter;

	timestep_ = 0.05;			  // Fokker-Planck equation parameters
	diff_coeff_ = 0.5;
	psi_ = 0.0;					  // trial wave function
	psi_new_ = 0.0;

	alpha_.zeros(D_);             // variational parameters
	r_.zeros(N_,D_);              // positions
	r_new_.zeros(N_,D_);
	qforce_.zeros(N_,D_);		  // quantum force
	qforce_new_.zeros(N_,D_);

	omega2_ = omega % omega;      // frequencies squared
	if(omega.n_elem != dim){
		cout << "WARNING: number of frequencies do not match dimension." << endl;
	}
}

// with importance sampling
void CBosonSystem::energy_minimization(int number_MC_cycles, vec alpha, string filename){

	double E = 0.0, E2 = 0.0, deltaE = 0.0, E_err;
	vec dE = zeros<vec>(D_);
	vec deltapsi = zeros<vec>(D_);
	vec dpsi = zeros<vec>(D_);
	vec dpsiE = zeros<vec>(D_);

	// randomly position particles
	random_initial_positions();

	// calculate initial wave function and quantum force
	psi_ = calc_trial_wavefunction(r_,alpha);
	qforce_ = calc_quantum_force(r_,alpha);

	// loop over monte carlo cycles
	for(int m = 0; m < number_MC_cycles; ++m){

		for(int i = 0; i < N_; ++i){

			// move ith particle
			random_new_position(i);

			// calculate new wave function and quantum force
			psi_new_ = calc_trial_wavefunction(r_new_,alpha);
			qforce_new_ = calc_quantum_force(r_new_,alpha);

			// Metropolis-Hastings test
			if(rand01() <= acceptance_ratio(i)){
				r_.row(i) = r_new_.row(i);
				qforce_.row(i) = qforce_new_.row(i);
				psi_ = psi_new_;
			}
		}

		// calculate energy and derivatives
		deltaE = calc_local_energy(r_,alpha);
		dpsi = calc_derivative_wavefunction(r_,alpha);
		E += deltaE;
		E2 += deltaE*deltaE;
		deltapsi += dpsi;
		dpsiE += dpsi*deltaE;
	}

	// calculate mean, variance, error
	E /= number_MC_cycles;
	E2 /= number_MC_cycles;
	dpsiE /= number_MC_cycles;
	deltapsi /= number_MC_cycles;
	E_err = sqrt((E2-E*E)/number_MC_cycles);
	dE = 2.0*(dpsiE-deltapsi*E);
}


// calculate trial wavefunction
double CBosonSystem::calc_trial_wavefunction(mat r, vec alpha){

	double psi = 1.0, sum = 0.0, rij;
	vec ri2;

	// no correlation part for a=0
	if(a_ > 0.0){
		for(int i = 0; i < N_; ++i){
			for(int j = 0; j < N_; ++j){
				if(i != j){

					rij = distance(r,i,j);

					if(rij <= a_) return 0.0;
					else psi *= 1.0-a_/rij;
				}
			}
		}
	}

	// elliptical harmonic oscillator part
	for(int i = 0; i < N_; ++i){

		ri2 = r.row(i) % r.row(i);
		sum += norm(dot(alpha,ri2));
	}
	psi *= exp(-sum);

	return psi;
}

// calculate local energy (see equations pdf in doc folder)
double CBosonSystem::calc_local_energy(mat r, vec alpha){

	double EL = 0.0, prefactor, rij;
	vec alpha2 = alpha%alpha;
	vec one = ones<vec>(D_);
	vec sum = zeros<vec>(D_);
	vec Ri = zeros<vec>(D_);
	vec Rj = zeros<vec>(D_);


	EL += N_*dot(alpha,one);

	for(int i = 0; i < N_; ++i){

		Ri = r.row(i); // check this is what i expect

		EL += 0.5*dot(omega2_, Ri%Ri);  // harmonic oscillator part
		EL -= 2.0*dot(alpha2, Ri%Ri);

		for(int j = 0; j < N_; ++j){
			if(i != j){

				Rj = r.row(j);
				rij = distance(r,i,j);
				prefactor = a_/(rij*rij*(rij-a_));

				sum += prefactor*(Ri-Rj);				
			}

		}

		EL -= 0.5*dot(sum,sum);
	}

	for(int i = 0; i < N_-1; ++i){
		for(int j = i+1; j < N_; ++j){

			Rj = r.row(j);
			rij = distance(r,i,j);

			prefactor = a_/(rij*rij*(rij-a_));

			EL += prefactor*(((3.0-D_)*rij+(D_-2.0)*a)/(rij-a_)+4.0*(alpha%Ri)*(Ri-Rj));
		}
	}

	return EL;
}

// calculate quantum force for each particle
mat CBosonSystem::calc_quantum_force(mat r, vec alpha){

	double prefactor, rij;
	vec Ri = zeros<vec>(D_);
	vec Rj = zeros<vec>(D_);
	mat qforce = zeros<mat>(N_,D_);

	for(int i = 0; i < N_; ++i){

		Ri = r.row(i);

		qforce.row(i) = -4.0*(alpha%Ri);

		for(int j = 0; j < N_; ++j){
			if(i != j){

				Rj = r.row(j);
				rij = distance(r,i,j);
				prefactor = a_/(rij*rij*(rij-a_));

				qforce.row(i) += 2.0*prefactor*(Ri-Rj);				
			}
		}
	}

	return qforce;
}

// calculate derivative of trial wavefunction wrt variational parameters
vec CBosonSystem::calc_derivative_wavefunction(mat r, vec alpha){

	double sum;
	vec dpsi = zeros<vec>(D_);

	for(int d = 0; d < D_; ++d){

		sum = 0.0;
		for(int i = 0; i < N_; ++i){
			sum += r(i,d)*r(i,d);
		}
		dpsi(d) = -sum;
	}
	dpsi = calc_trial_wavefunction(r,alpha)*dpsi;

	return dpsi;
}

// set random initial positions
void CBosonSystem::random_initial_positions(){

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < D_; ++j){
			r_(i,j) = sqrt(timestep_)*randnorm();
		}
	}
}

// get new position of ith particle
void CBosonSystem::random_new_position(int i){

	for(int d = 0; d < D_; ++d){
		r_new_(i,d) = r_(i,d)+diff_coeff_*qforce_(i,d)*timestep_+randnorm()*sqrt(timestep_);
	}
}

// acceptance ratio for Metropolis-Hastings algorithm
double CBosonSystem::acceptance_ratio(int i){

	double greens = 0.0;

	for(int d = 0; d < D_; ++d){
		greens += 0.5*(r_(i,d)-r_new_(i,d)(qforce_new_(i,d)+qforce_(i,d)));
		greens += 0.25*diff_coeff_*timestep_*(pow(qforce_(i,d),2.0)-pow(qforce_new_(i,d),2.0));
	}
	greens = exp(greens);

	return greens*psi_new_*psi_new_/(psi_*psi_);
}

// distance between ith and jth bosons
double CBosonSystem::distance(mat r, int i, int j){

	vec rij = r.row(i)-r.row(j);

	return norm(rij,2);
}

// uniform rng
double CBosonSystem::rand01(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}

// normal rng
double CBosonSystem::randnorm(){

	normal_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}

////////////////////////////////////////////////////////////////////////////////////////////

#include "boson_system.h"

CBosonSystem::CBosonSystem(int number_bosons, int max_variation, double position_step, double mass, double hard_core_diameter,
	                       double omega_xy, double omega_z, double alpha0, double alphaf, double beta0, double betaf){

	D_ = 3;
	N_ = number_bosons;
	max_ = max_variation;
	hbar_ = 1.0;
	m_ = mass;
	a_ = hard_core_diameter;
	step_ = position_step;

	psi_ = 0.0;
	psi_new_= 0.0;
	omega_xy_ = omega_xy;
	omega_z_ = omega_z;

	alpha_ = linspace(alpha0,alphaf,max_);
	beta_ = linspace(beta0,betaf,max_);
	r_.zeros(N_,D_);
	r_new_.zeros(N_,D_);
	E_.zeros(max_,max_);
	E_err_.zeros(max_,max_);
}

void CBosonSystem::montecarlo_sampling(int number_MC_cycles, string filename){

	// open file
	ofstream outfile;
	outfile.open(filename+"_"+to_string(max_)+"_"+to_string(number_MC_cycles)+".dat");
	outfile << "# alpha, beta, E, E_err" << endl;

	// heading
	printf("%10s %10s %10s %10s \n\n","alpha","beta","energy","error");

	// loop over various parameter values
	for(int a = 0; a < max_; ++a){
		for(int b = 0; b < max_; ++b){

			double E = 0.0, E2 = 0.0, DeltaE = 0.0, E_err;

			// randomly position particles
			random_initial_positions();

			// calculate initial wave function
			psi_ = calc_trial_wavefunction(r_,alpha_(a),beta_(b));

			// loop over monte carlo cycles
			for(int m = 0; m < number_MC_cycles; ++m){

				// propose new trial positions
				random_trial_positions();

				// calculate new trial wave function
				psi_new_ = calc_trial_wavefunction(r_new_,alpha_(a),beta_(b));

				// metropolis test
				if(rand01() < pow(psi_new_/psi_,2.0)){
					r_ = r_new_;
					psi_ = psi_new_;
					DeltaE = calc_local_energy(r_,alpha_(a),beta_(b));
				}
				E += DeltaE;
				E2 += DeltaE*DeltaE;
			}

			// calculate mean, variance, error
			E = E/number_MC_cycles;
			E2 = E2/number_MC_cycles;
			E_err = sqrt((E2-E*E)/number_MC_cycles);

			// store results
			E_(a,b) = E;
			E_err_(a,b) = E_err;

			// print results
			printf("%10.3lf %10.3lf %10.3lf %10.3lf \n",alpha_(a),beta_(b),E_(a,b),E_err_(a,b));

			outfile << left << setw(10) << setprecision(5) << alpha_(a);
			outfile << left << setw(10) << setprecision(5) << beta_(a);
			outfile << left << setw(10) << setprecision(5) << E_(a,b);
			outfile << left << setw(10) << setprecision(5) << E_err_(a,b) << endl;		}
	}

	outfile.close();
}

void CBosonSystem::random_initial_positions(){

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < D_; ++j){
			r_(i,j) = step_*(rand01()-0.5);
		}
	}
}

void CBosonSystem::random_trial_positions(){

	r_new_ = zeros<mat>(max_,max_);

	for(int i = 0; i < N_; ++i){
		for(int j = 0; j < D_; ++j){
			r_new_(i,j) = r_(i,j)+step_*(rand01()-0.5);
		}
	}
}

// distance between ith and jth particle
double CBosonSystem::distance(mat r, int i, int j){

	double distance = 0.0;

	for(int d = 0; d < D_; ++d){
		distance += pow(r(i,d)-r(j,d),2.0);
	}

	return sqrt(distance);
}


double CBosonSystem::calc_trial_wavefunction(mat r, double alpha, double beta){

	// correlation part
	double psi = 1.0, r_ij;

	if(a_ > 0.0){
		for(int i = 0; i < N_; i++){
			for(int j = 0; j < N_; j++){
				if(j != i){

					r_ij = distance(r,i,j);

					// wave function vanishes if distance between bosons <= a
					if(r_ij <= a_){
						return 0.0;
					}

					// correlation wave function
					else psi *= 1.0-a_/r_ij;				
				}
			}
		}	
	}


	// elliptical harmonic oscillator part
	double sum = 0.0;
	for(int i = 0; i < N_; ++i){
		sum += r(i,0)*r(i,0)+r(i,1)*r(i,1)+beta*r(i,2)*r(i,2);
	}
	psi *= exp(-alpha*sum);

	return psi;
}


double CBosonSystem::calc_local_energy(mat r, double alpha, double beta){

	// KINETIC
	double EL_kinetic = 0.0, del_i, del2_i, r_ij;

	// first derivatives
	for(int i = 0; i < N_; ++i){

		del_i = -2.0*alpha*(r_(i,0)+r_(i,1)+beta*r_(i,2));

		for(int j = 0; j < N_; ++j){
			if(j != i){
				r_ij = distance(r,i,j);
				del_i += (a_/(r_ij*r_ij*(r_ij-a_)))*(r(i,0)-r(j,0)+r(i,1)-r(j,1)+r(i,2)-r(j,2));
			}
		}

		EL_kinetic += del_i*del_i;
	}

	// second derivatives
	EL_kinetic += -2.0*alpha*N_*(2.0+beta);
	for(int i = 0; i < N_; ++i){

		for(int j = 0; j < N_; ++j){
			if(j != i){
				r_ij = distance(r,i,j);
				del2_i = a_/(r_ij*r_ij*(r_ij-a_));
				del2_i *= 3.0-(3.0*r_ij-2.0*a_)*pow((r(i,0)-r(j,0)+r(i,1)-r(j,1)+r(i,2)-r(j,2))/(r_ij*(r_ij-a_)),2.0);

			}
		}

		EL_kinetic += del2_i;
	}
	EL_kinetic *= -0.5*hbar_*hbar_/m_;


	// POTENTIAL
	double EL_potential = 0.0;

	// elliptical harmonic oscillator
	for(int i = 0; i < N_; ++i){
		EL_potential += omega_xy_*omega_xy_*(r(i,0)*r(i,0)+r(i,1)*r(i,1));
		EL_potential += omega_z_*omega_z_*r(i,2)*r(i,2);
	}
	EL_potential *= 0.5*m_;

	return EL_kinetic+EL_potential;
}


double CBosonSystem::rand01(){

	uniform_real_distribution<double> dist(0.0,1.0);
	return dist(rng_);
}