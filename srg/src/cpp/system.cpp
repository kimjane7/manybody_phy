#include "system.h"

CFermionSystem::CFermionSystem(){

	// default: 4-fermion pairing model
	N_ = 8;
	F_ = 4;
	particles_ = 4;
	sp_states_ = 4;
	d_ = 1.0;
	g_ = 0.5;

	generate_basis();
}

CFermionSystem::CFermionSystem(int particles, int sp_states, double d, double g){

	N_ = 2*sp_states;
	F_ = particles;
	particles_ = particles;
	sp_states_ = sp_states;
	d_ = d;
	g_ = g;

	generate_basis();

}

void CFermionSystem::generate_basis(){

	states_.set_size(N_,2);

	for(int i = 0; i < N_; ++i){

		// assign quantum number p
		states_(i,0) = floor(i/2.0);

		// assign spins
		if(i%2 == 0) states_(i,1) = 1;
		else states_(i,1) = -1;
	}
}

double CFermionSystem::h0(int q, int r){

	// return kinetic energy p*d
	if(q == r) return states_(q,0)*d_;
	else return 0.0;
}

double CFermionSystem::v(int q, int r, int s, int t){

	int p_q, p_r, p_s, p_t;
	int spin_q, spin_r, spin_s, spin_t;

	// retrieve quantum numbers 
	p_q = states_(q,0);
	p_r = states_(r,0);
	p_s = states_(s,0);
	p_t = states_(t,0);
	spin_q = states_(q,1);
	spin_r = states_(r,1);
	spin_s = states_(s,1);
	spin_t = states_(t,1);

	// return interaction energy
	double v = 0.0;
	if( (p_q == p_r) && (p_s == p_t) ){
		if( (spin_q == spin_s) && (spin_r == spin_t) ) v = -0.5*g_;
		if( (spin_q == spin_t) && (spin_r == spin_s) ) v = 0.5*g_;			
	}
	return v;
	
	/*
	if( (p_q == p_r) && (p_s == p_t) ){
		if( (spin_q == spin_r) || (spin_s == spin_t) ) return 0.0;
		else{
			if( (spin_q == spin_s) && (spin_r == spin_t) ) return -0.5*g_;
			if( (spin_q == spin_t) && (spin_r == spin_s) ) return 0.5*g_;			
		}
	}
	else return 0.0;
	*/
}

// fock operator
double CFermionSystem::f(int q, int r){

	double E = h0(q,r);
	for(int i = 0; i < F_; ++i) E += v(q,i,r,i);
	return E;
}

// redefined single-particle energies in HF basis (8.43)
double CFermionSystem::eps(int q, int r){

	if(q == r){
		double E = h0(q,q);
		for(int i = 0; i < F_; ++i) E += v(q,i,q,i);
		return E;
	}
	else return 0.0;
}

double CFermionSystem::eps(ivec holes, ivec parts){

	double E = 0.0;

	for(unsigned i = 0; i < holes.n_elem; ++i) E += f(holes(i),holes(i));
	for(unsigned a = 0; a < parts.n_elem; ++a) E -= f(parts(a),parts(a));

	return E;
}
