#pragma once
#include <Eigen/Dense>
#include <random>

constexpr double pi = 3.1415926535897932384626433832795;

// linear algebra aliases
using Array = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Real, 1, Eigen::Dynamic>;

// random number generators
extern std::mt19937_64 rng;
extern std::uniform_real_distribution<double> unif;
extern std::uniform_real_distribution<double> unifcentered;
extern std::normal_distribution<double> norm;

inline auto rand_unif(){
    return unif(rng);
}

inline auto rand_unifcentered(){
    return unifcentered(rng);
}

inline auto rand_norm(){
    return norm(rng);
}
