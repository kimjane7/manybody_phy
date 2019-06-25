#include "hamiltonian.h"

Hamiltonian::Hamiltonian(bool coulomb_int, VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    setup(coulomb_int, omega);
}

Hamiltonian::Hamiltonian(bool coulomb_int, double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    setup(coulomb_int, omega);

    a0_ = hard_core_diameter;
    if(a0_ > 0.0) bosons_ = true;
}

void Hamiltonian::setup(bool coulomb_int, VectorXd omega){

    coulomb_int_ = coulomb_int;
    bosons_ = false;
    omega_ = omega;

    // construct omega2 vector
    omega2_.resize(NQS_.M_);
    for(int i = 0; i < NQS_.M_; ++i){
        omega2_(i) = pow(omega_(i%NQS_.D_),2.0);
    }
}

void Hamiltonian::add_coulomb_interaction(){

    double R;

    // loop over unique pairs
    for(int p = 0; p < NQS_.P_-1; ++p){
        for(int q = 0; q < NQS_.P_; ++q){

            R = NQS_.distance(NQS_.x_,p,q);
            EL_ += 1.0/R;
        }
    }
}

void Hamiltonian::add_hardcore_interaction(int i){

    int p, d;
    double R, denom;

    p = floor(i/NQS_.D_);
    d = i%NQS_.D_;

    for(int q = 0; q < NQS_.P_; ++q){
        if(p != q){

            R = NQS_.distance(NQS_.x_,p,q);
            denom = R*R*(R-a0_);

            foo_ += a0_*(NQS_.x_(i)-NQS_.x_(NQS_.D_*q+d))/denom;
            EL_ -= (a0_/denom)*(pow((NQS_.x_(i)-NQS_.x_(NQS_.D_*q+d)),2.0)*(2.0*a0_-3.0*R)/denom+1.0); 
        }
    }

    EL_ -= foo_*foo_;
}

double Hamiltonian::calc_local_energy(){

    // precalculate B, sigmoidB, x2 for current x_
    VectorXd B = NQS_.calc_B(NQS_.x_);
    VectorXd sigmoidB = NQS_.calc_sigmoid(B);
    VectorXd x2 = (NQS_.x_.array()*NQS_.x_.array()).matrix();

    // calculate local energy
    double EL_ = NQS_.M_/NQS_.sigma2_ + omega2_.transpose()*x2;

    for(int j = 0; j < NQS_.N_; ++j){
        EL_ -= exp(-B(j))*(sigmoidB(j)*NQS_.W_.col(j)/NQS_.sigma2_).squaredNorm();
    }

    for(int i = 0; i < NQS_.M_; ++i){

        foo_ = (NQS_.a_(i)-NQS_.x_(i))/NQS_.sigma2_ + NQS_.W_.row(i)*sigmoidB;

        // hard core interaction for dilute bosons
        if(bosons_) add_hardcore_interaction(i);
    }

    EL_ = 0.5*EL_;

    // repulsive coulomb interaction
    if(coulomb_int_) add_coulomb_interaction();

    return EL_;    
}

double Hamiltonian::calc_psi(VectorXd x){

    double psi = NQS_.calc_psi(x);

    if(bosons_) psi *= calc_hardcore_jastrow_factor(x);
    if(coulomb_int_) psi *= calc_coulomb_jastrow_factor(x);

    return psi;
}

double Hamiltonian::calc_coulomb_jastrow_factor(VectorXd x){

    double R, jastrow = 0.0;

    // loop over unique pairs
    for(int p = 0; p < NQS_.P_-1; ++p){
        for(int q = 0; q < NQS_.P_; ++q){

            R = NQS_.distance(x,p,q);
            jastrow += R;
        }
    }

    return exp(jastrow);
}

double Hamiltonian::calc_hardcore_jastrow_factor(VectorXd x){

    double R, jastrow = 1.0;

    // loop over unique pairs
    for(int p = 0; p < NQS_.P_-1; ++p){
        for(int q = 0; q < NQS_.P_; ++q){

            R = NQS_.distance(x,p,q);
            jastrow *= (1.0-a0_/R);
        }
    }

    return jastrow;
}

VectorXd Hamiltonian::calc_gradient_logpsi(){

    int index;

    // precalculate B and sigmoidB for current x_
    VectorXd B = NQS_.calc_B(NQS_.x_);
    VectorXd sigmoidB = NQS_.calc_sigmoid(B);

    // calculate (gradient psi)/(psi) wrt all weights and biases
    VectorXd grad_logpsi(NQS_.M_+NQS_.N_+NQS_.M_*NQS_.N_);

    grad_logpsi(seq(0,NQS_.M_-1)) = (NQS_.x_-NQS_.a_)/NQS_.sigma2_;
    grad_logpsi(seq(NQS_.M_,NQS_.M_+NQS_.N_-1)) = sigmoidB;

    // outer product
    MatrixXd x_sigmoidBT = NQS_.x_*sigmoidB.transpose();
    for(int j = 0; j < NQS_.N_; ++j){

        index = NQS_.M_+NQS_.N_+j*NQS_.M_;
        grad_logpsi(seq(index,index+NQS_.M_-1)) = x_sigmoidBT.col(j)/NQS_.sigma2_;
    }

    return grad_logpsi;
}


VectorXd Hamiltonian::calc_quantum_force(int p, VectorXd x){

    int i;
    double denom, R;

    // precalculate B and sigmoidB for given x
    VectorXd B = NQS_.calc_B(x);
    VectorXd sigmoidB = NQS_.calc_sigmoid(B);

    // calculate quantum force for pth particle only
    VectorXd qforce(NQS_.D_);

    for(int d = 0; d < NQS_.D_; ++d){

        i = p*NQS_.D_+d;
        qforce(d) = (NQS_.a_(i)-x(i)+NQS_.W_.row(i)*sigmoidB)/NQS_.sigma2_;

        // interaction part
        if(a0_ > 0.0){
            for(int q = 0; q < NQS_.P_; ++q){
                if(p != q){

                    R = NQS_.distance(NQS_.x_,p,q);
                    denom = R*R*(R/a0_-1.0);
                    qforce(d) += (x(p*NQS_.D_+d)-x(q*NQS_.D_+d))/denom;
                }
            }
        }
    }

    return 2.0*qforce;
}

