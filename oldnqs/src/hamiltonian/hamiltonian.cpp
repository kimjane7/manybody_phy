#include "hamiltonian.h"

Hamiltonian::Hamiltonian(VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    setup(omega);
}

Hamiltonian::Hamiltonian(bool electrons, VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    setup(omega);
    electrons_ = electrons;
}

Hamiltonian::Hamiltonian(double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    setup(omega);
    a0_ = hard_core_diameter;
    if(a0_ > 0.0) bosons_ = true;
}

void Hamiltonian::setup(VectorXd omega){

    electrons_ = false;
    bosons_ = false;
    omega_ = omega;

    // construct omega2 vector
    omega2_.resize(NQS_.M_);
    for(int i = 0; i < NQS_.M_; ++i){
        omega2_(i) = pow(omega_(i%NQS_.D_),2.0);
    }
}

double Hamiltonian::calc_local_energy(){

    // placeholder
    VectorXd qforce(NQS_.D_);

    // precalculate B, sigmoidB, x2 for current x_
    VectorXd B = NQS_.calc_B(NQS_.x_);
    VectorXd sigmoidB = NQS_.calc_sigmoid(B);
    VectorXd x2 = (NQS_.x_.array()*NQS_.x_.array()).matrix();

    // harmonic oscillator part
    double EL = omega2_.transpose()*x2;

    // kinetic part
    EL += NQS_.M_/NQS_.sigma2_;
    for(int j = 0; j < NQS_.N_; ++j){
        EL -= exp(-B(j))*(sigmoidB(j)*NQS_.W_.col(j)/NQS_.sigma2_).squaredNorm();
    }

    for(int p = 0; p < NQS_.P_; ++p){

        if(bosons_) EL -= calc_hardcore_interaction(p);
        if(electrons_) EL -= (NQS_.D_-1)*calc_coulomb_interaction(p);

        qforce = calc_quantum_force(p,NQS_.x_);
        EL -= 0.25*qforce.squaredNorm();
    }

    // interaction potential part
    if(electrons_){
        for(int p = 0; p < NQS_.P_; ++p){
            EL += calc_coulomb_interaction(p);
        }
    }

    return 0.5*EL;   
}

double Hamiltonian::calc_psi(VectorXd x){

    double psi = NQS_.calc_psi(x);

    if(bosons_) psi *= calc_hardcore_jastrow_factor(x);
    if(electrons_) psi *= calc_coulomb_jastrow_factor(x);

    return psi;
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
    double R, denom;

    // precalculate B and sigmoidB for given x
    VectorXd B = NQS_.calc_B(x);
    VectorXd sigmoidB = NQS_.calc_sigmoid(B);

    // calculate quantum force for pth particle only
    VectorXd qforce(NQS_.D_);
    for(int d = 0; d < NQS_.D_; ++d){
        i = NQS_.D_*p+d;
        qforce(d) = NQS_.a_(i)-x(i)+NQS_.W_.row(i)*sigmoidB;
    }
    qforce /= NQS_.sigma2_;

    // interaction part
    if(bosons_){
        for(int q = 0; q < NQS_.P_; ++q){
            if(q != p){

                R = NQS_.distance(x,p,q);
                denom = R*R*(R/a0_-1.0);

                for(int d = 0; d < NQS_.D_; ++d){
                    qforce(d) += (x(NQS_.D_*p+d)-x(NQS_.D_*q+d))/denom;
                }
            }
        }
    }
    if(electrons_){
        for(int q = 0; q < NQS_.P_; ++q){
            if(q != p){

                R = NQS_.distance(x,p,q);

                for(int d = 0; d < NQS_.D_; ++d){
                    qforce(d) += (x(NQS_.D_*p+d)-x(NQS_.D_*q+d))/R;
                }                
            }
        }
    }

    return 2.0*qforce;
}

double Hamiltonian::calc_hardcore_interaction(int p){

    double R, denom, laplacian_J = 0.0;

    for(int q = 0; q < NQS_.P_; ++q){
        if(q != p){

            R = NQS_.distance(NQS_.x_,p,q);
            denom = R*R*(R/a0_-1.0)*(R-a0_);

            laplacian_J += ((2.0-NQS_.D_)*a0_ + (NQS_.D_-3.0)*R)/denom;
        }
    }

    return laplacian_J;
}

double Hamiltonian::calc_coulomb_interaction(int p){

    double R, laplacian_J = 0.0;

    for(int q = 0; q < NQS_.P_; ++q){
        if(q != p){

            R = NQS_.distance(NQS_.x_,p,q);

            laplacian_J += 1.0/R;
        }
    }

    return laplacian_J;
}

double Hamiltonian::calc_coulomb_jastrow_factor(VectorXd x){

    double R, jastrow = 0.0;

    // loop over unique pairs
    for(int p = 0; p < NQS_.P_-1; ++p){
        for(int q = p+1; q < NQS_.P_; ++q){

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
        for(int q = p+1; q < NQS_.P_; ++q){

            R = NQS_.distance(x,p,q);
            jastrow *= (1.0-a0_/R);
        }
    }

    return jastrow;
}