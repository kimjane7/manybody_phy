#pragma once
#include <random>
#include "../Eigen/Dense"
#include "../definitions.hpp"

class NeuralQuantumState{

public:
    
    int N_, M_;
    double psi_;
    Vector x_, alpha_;
}
