// In-medium SRG for the pairing model
#include "imsrg.h"

CIMSRG::CIMSRG(int dim1B, double d, double g, double smax, double ds, ivec holes, ivec parts){

	d_ = d;
	g_ = g;
	smax_ = smax;
	ds_ = ds;

	holes_ = holes;
	parts_ = parts;
	nholes_ = holes.n_elem;
	nparts_ = parts.n_elem;

	dim1B_ = dim1B;
	dim2B_ = dim1B*dim1B;
	basis1B_.set_size(dim1B);
	for(int i = 0; i < dim1B; ++i) basis1B_(i) = i;

	build_basis2B();
	build_ph_basis2B();
	build_occ1B();
	build_occ2B_1();
	build_ph_occ2B_1();
	build_occ2B_2();
	build_occ2B_3();
	build_hamiltonian();
	normal_order();

	max_ = 170;
	Omega0B_ = 0.0;
	Omega1B_.zeros(dim1B_,dim1B_);
	Omega2B_.zeros(dim2B_,dim2B_);
	build_prefactors();
}

void CIMSRG::build_basis2B(){

	basis2B_.set_size(dim2B_,2);
	irowvec state;
	int index = 0;

	for(int i = 0; i < nholes_; ++i){
		for(int j = 0; j < nholes_; ++j){
			basis2B_(index,0) = holes_(i);
			basis2B_(index,1) = holes_(j);
			state = basis2B_.row(index);
			index2B_[state] = index;
			index++;
		}
	}

	for(int i = 0; i < nholes_; ++i){
		for(int a = 0; a < nparts_; ++a){
			basis2B_(index,0) = holes_(i);
			basis2B_(index,1) = parts_(a);
			state = basis2B_.row(index);
			index2B_[state] = index;
			index++;
		}
	}

	for(int a = 0; a < nparts_; ++a){
		for(int i = 0; i < nholes_; ++i){
			basis2B_(index,0) = parts_(a);
			basis2B_(index,1) = holes_(i);
			state = basis2B_.row(index);
			index2B_[state] = index;
			index++;			
		}
	}

	for(int a = 0; a < nparts_; ++a){
		for(int b = 0; b < nparts_; ++b){
			basis2B_(index,0) = parts_(a);
			basis2B_(index,1) = parts_(b);
			state = basis2B_.row(index);
			index2B_[state] = index;
			index++;			
		}
	}
}

void CIMSRG::build_ph_basis2B(){

	ph_basis2B_.set_size(dim2B_,2);
	irowvec state;
	int index = 0;

	for(int i = 0; i < nholes_; ++i){
		for(int j = 0; j < nholes_; ++j){
			ph_basis2B_(index,0) = holes_(i);
			ph_basis2B_(index,1) = holes_(j);
			state = ph_basis2B_.row(index);
			ph_index2B_[state] = index;
			index++;
		}
	}

	for(int i = 0; i < nholes_; ++i){
		for(int a = 0; a < nparts_; ++a){
			ph_basis2B_(index,0) = holes_(i);
			ph_basis2B_(index,1) = parts_(a);
			state = ph_basis2B_.row(index);
			ph_index2B_[state] = index;
			index++;
		}
	}

	for(int a = 0; a < nparts_; ++a){
		for(int i = 0; i < nholes_; ++i){
			ph_basis2B_(index,0) = parts_(a);
			ph_basis2B_(index,1) = holes_(i);
			state = ph_basis2B_.row(index);
			ph_index2B_[state] = index;
			index++;			
		}
	}

	for(int a = 0; a < nparts_; ++a){
		for(int b = 0; b < nparts_; ++b){
			ph_basis2B_(index,0) = parts_(a);
			ph_basis2B_(index,1) = parts_(b);
			state = ph_basis2B_.row(index);
			ph_index2B_[state] = index;
			index++;			
		}
	}
}

void CIMSRG::build_occ1B(){

	occ1B_.zeros(dim1B_);
	for(int i = 0; i < nholes_; ++i) occ1B_(holes_(i)) = 1;
}

// n_p - n_q
void CIMSRG::build_occ2B_1(){

	occ2B_1_.zeros(dim2B_,dim2B_);
	for(int p = 0; p < dim2B_; ++p){
		occ2B_1_(p,p) = occ1B_(basis2B_(p,0))-occ1B_(basis2B_(p,1));
	}
}

// 1 - n_p - n_q
void CIMSRG::build_occ2B_2(){

	occ2B_2_.zeros(dim2B_,dim2B_);
	for(int p = 0; p < dim2B_; ++p){
		occ2B_2_(p,p) = 1-occ1B_(basis2B_(p,0))-occ1B_(basis2B_(p,1));
	}	
}

// n_p * n_q
void CIMSRG::build_occ2B_3(){

	occ2B_3_.zeros(dim2B_,dim2B_);
	for(int p = 0; p < dim2B_; ++p){
		occ2B_3_(p,p) = occ1B_(basis2B_(p,0))*occ1B_(basis2B_(p,1));
	}	
}

void CIMSRG::build_ph_occ2B_1(){

	ph_occ2B_1_.zeros(dim2B_,dim2B_);
	for(int p = 0; p < dim2B_; ++p){
		ph_occ2B_1_(p,p) = occ1B_(ph_basis2B_(p,0))-occ1B_(ph_basis2B_(p,1));
	}	
}

void CIMSRG::build_hamiltonian(){

	int i, j, k, l;

	// one-body
	H1B_.zeros(dim1B_,dim1B_);
	for(int p = 0; p < dim1B_; ++p){
		i = basis1B_(p);
		H1B_(i,i) = d_*floor(i/2.0);
	}

	// two-body
	H2B_.zeros(dim2B_,dim2B_);
	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);

		if((i%2 == 0) && (j == i+1)){

			for(int q = 0; q < dim2B_; ++q){

				k = basis2B_(q,0);
				l = basis2B_(q,1);

				if((k%2 == 0) && (l == k+1)){

					H2B_(index2B_[{i,j}],index2B_[{k,l}]) = -0.5*g_;
					H2B_(index2B_[{i,j}],index2B_[{l,k}]) = 0.5*g_;
					H2B_(index2B_[{j,i}],index2B_[{k,l}]) = 0.5*g_;
					H2B_(index2B_[{j,i}],index2B_[{l,k}]) = -0.5*g_;	
				}
			}
		}
	}
}

void CIMSRG::normal_order(){

	int index1, index2;

	// zero-body
	E_i_ = 0.0;
	for(int i = 0; i < nholes_; ++i){
		E_i_ += H1B_(holes_(i),holes_(i));
		for(int j = 0; j < nholes_; ++j){
			index1 = index2B_[{holes_(i),holes_(j)}];
			E_i_ += 0.5*H2B_(index1,index1);
		}
	}

	// one-body
	f_i_ = H1B_;
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){
				index1 = index2B_[{p,holes_(i)}];
				index2 = index2B_[{q,holes_(i)}];
				f_i_(p,q) += H2B_(index1,index2);
			}
		}
	}

	// two-body
	Gamma_i_ = H2B_;
}

double CIMSRG::fod_norm(){

	double norm = 0.0;

	for(int a = 0; a < nparts_; ++a){
		for(int i = 0; i < nholes_; ++i){
			norm += f_(parts_(a),holes_(i))*f_(parts_(a),holes_(i));
			norm += f_(holes_(i),parts_(a))*f_(holes_(i),parts_(a));
		}
	}

	return sqrt(norm);
}

double CIMSRG::Gammaod_norm(){

	double norm = 0.0;
	int index1, index2;

	for(int a = 0; a < nparts_; ++a){
		for(int b = 0; b < nparts_; ++b){
			for(int i = 0; i < nholes_; ++i){
				for(int j = 0; j < nholes_; ++j){
					index1 = index2B_[{parts_(a),parts_(b)}];
					index2 = index2B_[{holes_(i),holes_(j)}];
					norm += Gamma_(index1,index2)*Gamma_(index1,index2);
					norm += Gamma_(index2,index1)*Gamma_(index2,index1);
				}
			}
		}
	}

	return sqrt(norm);
}

mat CIMSRG::ph_transform2B(mat matrix2B){

	int i, j, k, l;
	mat ph_matrix2B = zeros<mat>(dim2B_,dim2B_);

	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);

		for(int q = 0; q < dim2B_; ++q){

			k = basis2B_(q,0);
			l = basis2B_(q,1);
			
			ph_matrix2B(p,q) -= matrix2B(index2B_[{i,j}],index2B_[{k,l}]);
		}
	}

	return ph_matrix2B;
}

mat CIMSRG::inverse_ph_transform2B(mat ph_matrix2B){

	int i, j, k, l;
	mat matrix2B = zeros<mat>(dim2B_,dim2B_);

	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);

		for(int q = 0; q < dim2B_; ++q){

			k = basis2B_(q,0);
			l = basis2B_(q,1);
			
			matrix2B(p,q) -= ph_matrix2B(ph_index2B_[{i,j}],ph_index2B_[{k,l}]);
		}
	}

	return matrix2B;	
}

mat CIMSRG::commutator(mat A, mat B){

	return A*B-B*A;
}

void CIMSRG::calc_eta_wegner(){

	int index1, index2, index3;
	mat fd, fod, Gammad, Gammaod, GammaGamma, GammaGammaT;
	mat ph_Gammad, ph_Gammaod, ph_GammaGamma;

	// split into diagonal and off-diagonal parts
	fd.zeros(size(f_));
	fod.zeros(size(f_));
	Gammad.zeros(size(Gamma_));
	Gammaod.zeros(size(Gamma_));

	for(int a = 0; a < nparts_; ++a){
		for(int i = 0; i < nholes_; ++i){
			fod(parts_(a),holes_(i)) = f_(parts_(a),holes_(i));
			fod(holes_(i),parts_(a)) = f_(holes_(i),parts_(a));
		}
	}
	fd = f_-fod;

	for(int a = 0; a < nparts_; ++a){
		for(int b = 0; b < nparts_; ++b){
			for(int i = 0; i < nholes_; ++i){
				for(int j = 0; j < nholes_; ++j){

					index1 = index2B_[{parts_(a),parts_(b)}];
					index2 = index2B_[{holes_(i),holes_(j)}];

					Gammaod(index1,index2) = Gamma_(index1,index2);
					Gammaod(index2,index1) = Gamma_(index2,index1);
				}
			}
		}
	}
	Gammad = Gamma_-Gammaod;

	// ONE-BODY PART
	// 1B-1B (checked)
	eta1B_ = commutator(fd, fod);

	// 1B-2B (checked)
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){
				for(int a = 0; a < nparts_; ++a){

					index1 = index2B_[{parts_(a),p}];
					index2 = index2B_[{holes_(i),q}];

					eta1B_(p,q) += fd(holes_(i),parts_(a))*Gammaod(index1,index2)-fod(holes_(i),parts_(a))*Gammad(index1,index2);

					index1 = index2B_[{holes_(i),p}];
					index2 = index2B_[{parts_(a),q}];

					eta1B_(p,q) += fod(parts_(a),holes_(i))*Gammad(index1,index2)-fd(parts_(a),holes_(i))*Gammaod(index1,index2);
				}
			}
		}
	}

	// 2B-2B
	GammaGamma = Gammad*occ2B_2_*Gammaod;
	GammaGammaT = GammaGamma.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){

				index1 = index2B_[{holes_(i),p}];
				index2 = index2B_[{holes_(i),q}];

				eta1B_(p,q) += 0.5*(GammaGamma(index1,index2)+GammaGammaT(index1,index2));
			}
		}
	}

	GammaGamma = Gammad*occ2B_3_*Gammaod;
	GammaGammaT = GammaGamma.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){

				index1 = index2B_[{r,p}];
				index2 = index2B_[{r,q}];

				eta1B_(p,q) += 0.5*(GammaGamma(index1,index2)+GammaGammaT(index1,index2));
			}
		}
	}

	// TWO-BODY PART
	// 1B-2B interaction
	eta2B_.zeros(size(Gamma_));
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){
				for(int s = 0; s < dim1B_; ++s){

					index1 = index2B_[{p,q}];
					index2 = index2B_[{r,s}];

					for(int t = 0; t < dim1B_; ++t){

						index3 = index2B_[{t,q}];
						eta2B_(index1,index2) += 0.25*(fd(p,t)*Gammaod(index3,index2)-fod(p,t)*Gammad(index3,index2));
						
						index3 = index2B_[{p,t}];
						eta2B_(index1,index2) -= 0.25*(fd(q,t)*Gammaod(index3,index2)-fod(p,t)*Gammad(index3,index2));					

						index3 = index2B_[{t,s}];
						eta2B_(index1,index2) -= 0.25*(fd(t,r)*Gammaod(index1,index3)-fod(t,r)*Gammad(index1,index3));	

						index3 = index2B_[{t,r}];
						eta2B_(index1,index2) += 0.25*(fd(t,s)*Gammaod(index1,index3)-fod(t,s)*Gammad(index1,index3));	
					}
				}
			}
		}
	}

	// 2B-2B interaction (factor 1/4 missing and transpose sign)
	GammaGamma = Gammad*occ2B_2_*Gammaod;
	GammaGammaT = GammaGamma.t();
	eta2B_ += 0.125*(GammaGamma+GammaGammaT);

	// transform to particle-hole representation
	ph_Gammad = ph_transform2B(Gammad);
	ph_Gammaod = ph_transform2B(Gammaod);
	ph_GammaGamma = ph_Gammad*ph_occ2B_1_*ph_Gammaod;

	// transform back
	GammaGamma = inverse_ph_transform2B(ph_GammaGamma);

	// commutator/antisymmetrization
	int i, j, k, l;
	mat work = zeros<mat>(size(GammaGamma));
	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);

		for(int q = 0; q < dim2B_; ++q){

			k = basis2B_(q,0);
			l = basis2B_(q,1);

			work(p,q) += (GammaGamma(index2B_[{j,i}],q)+GammaGamma(p,index2B_[{l,k}])
				          -GammaGamma(p,q)-GammaGamma(index2B_[{j,i}],index2B_[{l,k}]));
		}
	}
	GammaGamma = 0.25*work;
	eta2B_ += GammaGamma;
}

void CIMSRG::calc_derivatives(){

	int index1, index2, index3;
	mat etaGamma, etaGammaT;
	mat ph_eta, ph_Gamma, ph_etaGamma;

	// calculate generator
	calc_eta_wegner();

	// ZERO-BODY PART
	// 1B-1B
	dE_ = 0.0;
	for(int i = 0; i < nholes_; ++i){
		for(int a = 0; a < nparts_; ++a){
			dE_ += (eta1B_(holes_(i),parts_(a))*f_(parts_(a),holes_(i))
			       - eta1B_(parts_(a),holes_(i))*f_(holes_(i),parts_(a)));
		}
	}

	// 2B-2B
	for(int i = 0; i < nholes_; ++i){
		for(int j = 0; j < nholes_; ++j){
			for(int a = 0; a < nparts_; ++a){
				for(int b = 0; b <nparts_; ++b){

					index1 = index2B_[{holes_(i),holes_(j)}];
					index2 = index2B_[{parts_(a),parts_(b)}];

					dE_ += 0.5*eta2B_(index1,index2)*Gamma_(index2,index1);
				}
			}
		}
	}

	// ONE-BODY PART
	// 1B-1B
	df_ = commutator(eta1B_,f_);

	// 1B-2B
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){
				for(int a = 0; a < nparts_; ++a){

					index1 = index2B_[{parts_(a),p}];
					index2 = index2B_[{holes_(i),q}];

					df_(p,q) += (eta1B_(holes_(i),parts_(a))*Gamma_(index1,index2)-f_(holes_(i),parts_(a))*eta2B_(index1,index2));

					index1 = index2B_[{holes_(i),p}];
					index2 = index2B_[{parts_(a),q}];

					df_(p,q) += (f_(parts_(a),holes_(i))*eta2B_(index1,index2)-eta1B_(parts_(a),holes_(i))*Gamma_(index1,index2));
				}
			}
		}
	}

	// 2B-2B
	etaGamma = eta2B_*occ2B_2_*Gamma_;
	etaGammaT = etaGamma.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){

				index1 = index2B_[{i,p}];
				index2 = index2B_[{i,q}];	

				df_(p,q) += 0.5*(etaGamma(index1,index2)+etaGammaT(index1,index2));		
			}
		}
	}

	etaGamma = eta2B_*occ2B_3_*Gamma_;
	etaGammaT = etaGamma.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){

				index1 = index2B_[{r,p}];
				index2 = index2B_[{r,q}];

				df_(p,q) += 0.5*(etaGamma(index1,index2)+etaGammaT(index1,index2));
			}
		}
	}

	// TWO-BODY PART
	dGamma_ = zeros<mat>(size(Gamma_));
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){
				for(int s = 0; s < dim1B_; ++s){

					index1 = index2B_[{p,q}];
					index2 = index2B_[{r,s}];

					for(int t = 0; t < dim1B_; ++t){

						index3 = index2B_[{t,q}];
						dGamma_(index1,index2) += (eta1B_(p,t)*Gamma_(index3,index2)-f_(p,t)*eta2B_(index3,index2));

						index3 = index2B_[{p,t}];
						dGamma_(index1,index2) += (eta1B_(q,t)*Gamma_(index3,index2)-f_(q,t)*eta2B_(index3,index2));

						index3 = index2B_[{t,s}];
						dGamma_(index1,index2) += (f_(t,r)*eta2B_(index1,index3)-eta1B_(t,r)*Gamma_(index1,index3));					

						index3 = index2B_[{r,t}];
						dGamma_(index1,index2) += (f_(t,s)*eta2B_(index1,index3)-eta1B_(t,s)*Gamma_(index1,index3));
					}
				}
			}
		}
	}

	etaGamma = eta2B_*occ2B_2_*Gamma_;
	dGamma_ += 0.5*(etaGamma+trans(etaGamma));

	// transform matrices to ph representation
	ph_eta = ph_transform2B(eta2B_);
	ph_Gamma = ph_transform2B(Gamma_);
	ph_etaGamma = ph_eta*ph_occ2B_1_*ph_Gamma;

	// transform back
	etaGamma = inverse_ph_transform2B(ph_etaGamma);

	// antisymmetrization
	int i, j, k, l;
	mat work = zeros<mat>(size(etaGamma));
	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);
		index1 = index2B_[{j,i}];

		for(int q = 0; q < dim2B_; ++q){

			k = basis2B_(q,0);
			l = basis2B_(q,1);
			index2 = index2B_[{l,k}];

			work(p,q) += (etaGamma(index1,q)+etaGamma(p,index2)-etaGamma(p,q)-etaGamma(index1,index2));
		}
	}
	etaGamma = work;
	dGamma_ += etaGamma;
}

void CIMSRG::RK2_imsrg(){

	double E_k1, E_k2;
	mat f_k1, f_k2;
	mat Gamma_k1, Gamma_k2;

	// store old values
	double E = E_;
	mat f = f_;
	mat Gamma = Gamma_;

	// K1
	calc_derivatives();
	E_k1 = ds_*dE_;
	f_k1 = ds_*df_;
	Gamma_k1 = ds_*dGamma_;

	// K2
	E_ = E+0.5*E_k1;
	f_ = f+0.5*f_k1;
	Gamma_ = Gamma+0.5*Gamma_k1;
	calc_derivatives();
	E_k2 = ds_*dE_;
	f_k2 = ds_*df_;
	Gamma_k2 = ds_*dGamma_;

	// update
	E_ = E+E_k2;
	f_ = f+f_k2;
	Gamma_ = Gamma+Gamma_k2;
}

void CIMSRG::imsrg(vec snapshots, string filename){

	int k = 0;

	// initial values
	E_ = E_i_;
	f_ = f_i_;
	Gamma_ = Gamma_i_;

	ofstream outfile;
	outfile.open(filename);
	outfile << "# energy spacing = " << d_ << endl;
	outfile << "# interaction strength = " << g_ << endl;
	outfile << "# flow parameter s, zero-body component E, f_od norm, Gamma_od norm" << endl;

	for(double s = 0; s <= smax_; s += ds_){

		printf("s = %8.3f   dE = %8.5f   E = %8.5f\n", s, dE_, E_);

		RK2_imsrg();

		// write snapshots to file
		if(fabs(s-snapshots(k)) < 0.5*ds_){
			outfile << s << "\t" << E_ << "\t" << fod_norm() << "\t" << Gammaod_norm() << endl;
			++k;
		}
	}
}

double CIMSRG::imsrg(){

	// initial values
	E_ = E_i_;
	f_ = f_i_;
	Gamma_ = Gamma_i_;

	for(double s = 0; s <= smax_; s += ds_) RK2_imsrg();
	return E_;
}





void CIMSRG::build_prefactors(){

	// store factorial values
	factorial_.set_size(max_);
	for(int n = 0; n < max_; ++n) factorial_(n) = factorial(n);

	// store Bernoulli numbers
	vec B(max_);
	B(0) = 1.0;

	// set all odd Bernoulli numbers
	B(1) = -0.5; 
	for(int n = 3; n < max_; n += 2) B(n) = 0.0;

	// use recursive relation to calculate even Bernoulli numbers
	for(int n = 2; n < max_; n += 2){
		B(n) = 0.0;
		for(int k = 0; k < n; ++k) B(n) -= binomial_coeff(n+1,k)*B(k)/(n+1);
	}
	
	// calculate prefactors
	prefactor_.set_size(max_);
	for(int n = 0; n < max_; ++n) prefactor_(n) = B(n)/factorial_(n);
}

double CIMSRG::factorial(int n){

	if(n > 0){
		double logfactorial = 0.0;
		for(int i = 1; i <= n; ++i) logfactorial += log((double) i);
		return exp(logfactorial);
	}
	else return 1.0;
}

double CIMSRG::binomial_coeff(int n, int k){

	return factorial(n)/(factorial(k)*factorial(n-k));
}

void CIMSRG::commutator(mat A1B, mat A2B, mat B1B, mat B2B, double& C0B, mat& C1B, mat& C2B){

	int index1, index2, index3;

	// ZERO-BODY PART
	C0B = 0.0;

	// 1B-1B (checked)
	for(int i = 0; i < nholes_; ++i){
		for(int a = 0; a < nparts_; ++a){
			C0B += (A1B(holes_(i),parts_(a))*B1B(parts_(a),holes_(i))-A1B(parts_(a),holes_(i))*B1B(holes_(i),parts_(a)));
		}
	}

	// 2B-2B (checked)
	for(int i = 0; i < nholes_; ++i){
		for(int j = 0; j < nholes_; ++j){
			for(int a = 0; a < nparts_; ++a){
				for(int b = 0; b < nparts_; ++b){

					index1 = index2B_[{holes_(i),holes_(j)}];
					index2 = index2B_[{parts_(a),parts_(b)}];

					C0B += 0.25*(A2B(index1,index2)*B2B(index2,index1)-B2B(index1,index2)*A2B(index2,index1));
				}
			}
		}
	}

	// ONE-BODY PART
	// 1B-1B (checked)
	C1B = commutator(A1B,B1B);

	// 1B-2B (checked)
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){
				for(int a = 0; a < nparts_; ++a){

					index1 = index2B_[{parts_(a),p}];
					index2 = index2B_[{holes_(i),q}];

					C1B(p,q) += A1B(holes_(i),parts_(a))*B2B(index1,index2)-B1B(holes_(i),parts_(a))*A2B(index1,index2);
					
					index1 = index2B_[{holes_(i),p}];
					index2 = index2B_[{parts_(a),q}];	

					C1B(p,q) += B1B(parts_(a),holes_(i))*A2B(index1,index2)-A1B(parts_(a),holes_(i))*B2B(index1,index2);		
				}
			}
		}
	}

	// 2B-2B (checked)
	mat AB = A2B*occ2B_2_*B2B;
	mat ABT = AB.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int i = 0; i < nholes_; ++i){

				index1 = index2B_[{holes_(i),p}];
				index2 = index2B_[{holes_(i),q}];	

				C1B(p,q) += 0.5*(AB(index1,index2)+ABT(index1,index2));		
			}
		}
	}

	AB = A2B*occ2B_3_*B2B;
	ABT = AB.t();
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){

				index1 = index2B_[{r,p}];
				index2 = index2B_[{r,q}];

				C1B(p,q) += 0.5*(AB(index1,index2)+ABT(index1,index2));
			}
		}
	}

	// TWO-BODY PART
	// 1B-2B (checked, factor of 1/4 was missing)
	C2B = zeros<mat>(size(A2B));
	for(int p = 0; p < dim1B_; ++p){
		for(int q = 0; q < dim1B_; ++q){
			for(int r = 0; r < dim1B_; ++r){
				for(int s = 0; s < dim1B_; ++s){

					index1 = index2B_[{p,q}];
					index2 = index2B_[{r,s}];

					for(int t = 0; t < dim1B_; ++t){

						index3 = index2B_[{t,q}];
						C2B(index1,index2) += 0.25*(A1B(p,t)*B2B(index3,index2)-B1B(p,t)*A2B(index3,index2));

						index3 = index2B_[{t,p}];
						C2B(index1,index2) -= 0.25*(A1B(q,t)*B2B(index3,index2)-B1B(q,t)*A2B(index3,index2));

						index3 = index2B_[{t,s}];
						C2B(index1,index2) -= 0.25*(A1B(t,r)*B2B(index1,index3)-B1B(t,r)*A2B(index1,index3));					

						index3 = index2B_[{t,r}];
						C2B(index1,index2) += 0.25*(A1B(t,s)*B2B(index1,index3)-B1B(t,s)*A2B(index1,index3));
					}
				}
			}
		}
	}

	// 2B-2B (checked, don't understand antisymm entirely)
	AB = A2B*occ2B_2_*B2B;
	C2B += 0.125*(AB+trans(AB));

	// transform matrices to ph representation
	mat ph_A = ph_transform2B(A2B);
	mat ph_B = ph_transform2B(B2B);
	mat ph_AB = ph_A*ph_occ2B_1_*ph_B;

	// transform back
	AB = inverse_ph_transform2B(ph_AB);

	// antisymmetrization
	int i, j, k, l;
	mat work = zeros<mat>(size(AB));
	for(int p = 0; p < dim2B_; ++p){

		i = basis2B_(p,0);
		j = basis2B_(p,1);
		index1 = index2B_[{j,i}];

		for(int q = 0; q < dim2B_; ++q){

			k = basis2B_(q,0);
			l = basis2B_(q,1);
			index2 = index2B_[{l,k}];

			work(p,q) += (AB(index1,q)+AB(p,index2)-AB(p,q)-AB(index1,index2));
		}
	}
	AB = 0.25*work;
	C2B += AB;
}

void CIMSRG::calc_dOmega(){

	cout << "calculating eta...";
	calc_eta_wegner();
	cout << " done" << endl;

	// for storing commutator outputs
	double C0B;
	mat C1B, C2B;

	// k=0 term 
	dOmega0B_ = 0.0;
	dOmega1B_ = eta1B_;
	dOmega2B_ = eta2B_;

	eta1B_.print();

	// only one non-zero odd Bernoulli number B(k=1) = -0.5
	commutator(Omega1B_,Omega2B_,eta1B_,eta2B_,C0B,C1B,C2B);
	dOmega0B_ += -0.5*C0B;
	dOmega1B_ += -0.5*C1B;
	dOmega2B_ += -0.5*C2B;


	int k = 2;
	do{
		// calculate nested commutator
		commutator(Omega1B_,Omega2B_,C1B,C2B,C0B,C1B,C2B);

		// add remaining even terms
		if(k%2 == 0){
			dOmega0B_ += prefactor_(k)*C0B;
			dOmega1B_ += prefactor_(k)*C1B;
			dOmega2B_ += prefactor_(k)*C2B;			
		}

		cout << k << "\t" << C0B << endl;

		k++;

	}while(k < max_);
}

void CIMSRG::calc_hamiltonian(){

	double C0B;
	mat C1B, C2B;

	// k=0 term
	E_ = E_i_;
	f_ = f_i_;
	Gamma_ = Gamma_i_;

	// k=1 term
	commutator(Omega1B_,Omega2B_,f_i_,Gamma_i_,C0B,C1B,C2B);
	E_ += C0B;
	f_ += C1B;
	Gamma_ += C2B;

	int k = 2;
	do{
		// calculate nested commutator
		commutator(Omega1B_,Omega2B_,C1B,C2B,C0B,C1B,C2B);

		// add remaining even terms
		if(k%2 == 0){
			E_ += C0B/factorial_(k);
			f_ += C1B/factorial_(k);
			Gamma_ += C2B/factorial_(k);			
		}

		k++;

	}while(k < max_);
}

void CIMSRG::magnus(vec snapshots, string filename){

	int k = 0;

	// initial values
	E_ = E_i_;
	f_ = f_i_;
	Gamma_ = Gamma_i_;

	ofstream outfile;
	outfile.open(filename);
	outfile << "# energy spacing = " << d_ << endl;
	outfile << "# interaction strength = " << g_ << endl;
	outfile << "# flow parameter s, zero-body component E, f_od norm, Gamma_od norm" << endl;

	for(double s = 0; s <= smax_; s += ds_){

		printf("s = %8.3f \n", s);

		// write snapshots to file
		if(fabs(s-snapshots(k)) < 0.5*ds_){

			calc_hamiltonian();
			outfile << s << "\t" << E_ << "\t" << fod_norm() << "\t" << Gammaod_norm() << endl;
			k++;
		}


		// forward euler
		calc_dOmega();
		Omega0B_ += ds_*dOmega0B_;
		Omega1B_ += ds_*dOmega1B_;
		Omega2B_ += ds_*dOmega2B_;
	}

	outfile.close();
}



int main(int argc, char *argv[]){

	double d = atof(argv[1]);
	double g = atof(argv[2]);
	double smax = 10.0;
	double ds = atof(argv[3]);

	ivec holes = {0,1,2,3};
	ivec parts = {4,5,6,7};
	CIMSRG PairingModel(8, d, g, smax, ds, holes, parts);
	
	vec snapshots(38);
	snapshots(0) = 0.0;

	int n = 1;
	for(int k = 0; k < 4; ++k){
		for(int i = 1; i < 10; ++i){
			snapshots(n) = ds*i*pow(10.0,k);
			n++;
		}
	}
	snapshots(n) = smax;

	PairingModel.imsrg(snapshots, "pairing_imsrg.dat");
	//PairingModel.magnus(snapshots, "pairing_imsrg_magnus.dat");
	



	return 0;
}