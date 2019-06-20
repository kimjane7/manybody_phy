#include "hamiltonian.h"

Hamiltonian::Hamiltonian(double hard_core_diameter, VectorXd omega, NeuralQuantumState &NQS):
    NQS_(NQS){

    a0_ = hard_core_diameter;
    omega_ = omega;

    // construct omega2 vector
    VectorXd omega2_(NQS_.M_);
    for(int i = 0; i < NQS_.M_; ++i){
        omega2_(i) = pow(omega_(i%NQS_.D_),2.0);
    }
}

double Hamiltonian::calc_local_energy(){

    int p, d;
    double foo, denom, R;

    // precalculate B and sigmoidB for current x_
    VectorXd B = NQS_.calc_B();
    VectorXd sigmoidB = NQS_.calc_sigmoidB(B);

    // calculate local energy
    double EL = NQS_.M_/NQS_.sigma2_ + omega2_.dot(NQS_.x_.cwiseProduct(NQS_.x_));

    for(int j = 0; j < NQS_.N_; ++j){

        EL -= exp(-B(j))*(sigmoidB(j)*NQS_.W_.col(j)/NQS_.sigma2_).squaredNorm();
    }

    for(int i = 0; i < NQS_.M_; ++i){

        foo = (NQS_.a_(i)-NQS_.x_(i))/NQS_.sigma2_ + NQS_.W_.row(i)*sigmoidB;

        // weak interaction
        if(a0_ > 0.0){

            p = floor(i/NQS_.D_);
            d = i%NQS_.D_;

            for(int q = 0; q < NQS_.P_; ++q){
                if(p != q){

                    R = NQS_.distance(p,q);
                    denom = R*R*(R-a0_);

                    foo += a0_*(NQS_.x_(i)-NQS_.x_(NQS_.D_*q+d))/denom;
                    EL -= (a0_/denom)*(pow((NQS_.x_(i)-NQS_.x_(NQS_.D_*q+d)),2.0)*(2.0*a0_-3.0*R)/denom+1.0); 
                }
            }

            EL -= foo*foo;
        }
    }

    return 0.5*EL;    
}

VectorXd Hamiltonian::calc_gradient_logpsi(){

    int index;

    // precalculate B and sigmoidB for current x_
    VectorXd B = NQS_.calc_B();
    VectorXd sigmoidB = NQS_.calc_sigmoidB(B);

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



VectorXd Hamiltonian::calc_quantum_force(int p){

    int i;
    double denom, R;

    // precalculate B and sigmoidB for current x_
    VectorXd B = NQS_.calc_B();
    VectorXd sigmoidB = NQS_.calc_sigmoidB(B);

    // calculate quantum force for pth particle only
    VectorXd qforce(NQS_.D_);

    for(int d = 0; d < NQS_.D_; ++d){

        i = p*NQS_.D_+d;
        qforce(d) = (NQS_.a_(i)-NQS_.x_(i)+NQS_.W_.row(i)*sigmoidB)/NQS_.sigma2_;

        // interaction part
        if(a0_ > 0.0){
            for(int q = 0; q < NQS_.P_; ++q){
                if(p != q){

                    R = NQS_.distance(p,q);
                    denom = R*R*(R/a0_-1.0);
                    qforce(d) += (NQS_.x_(p*NQS_.D_+d)-NQS_.x_(q*NQS_.D_+d))/denom;
                }
            }
        }
    }

    return 2.0*qforce;
}

VectorXd Hamiltonian::calc_quantum_force(int p, VectorXd x){

    int i;
    double denom, R;

    // precalculate B and sigmoidB for given x
    VectorXd B = NQS_.calc_B(x);
    VectorXd sigmoidB = NQS_.calc_sigmoidB(B);

    // calculate quantum force for pth particle only
    VectorXd qforce(NQS_.D_);

    for(int d = 0; d < NQS_.D_; ++d){

        i = p*NQS_.D_+d;
        qforce(d) = (NQS_.a_(i)-x(i)+NQS_.W_.row(i)*sigmoidB)/NQS_.sigma2_;

        // interaction part
        if(a0_ > 0.0){
            for(int q = 0; q < NQS_.P_; ++q){
                if(p != q){

                    R = NQS_.distance(p,q);
                    denom = R*R*(R/a0_-1.0);
                    qforce(d) += (x(p*NQS_.D_+d)-x(q*NQS_.D_+d))/denom;
                }
            }
        }
    }

    return 2.0*qforce;
}

