#include "definitions.hpp"

std::mt19937_64 rng(12345);
std::uniform_real_distribution<double> unif(0.0,1.0);
std::uniform_real_distribution<double> unifcentered(-0.5,0.5);
std::normal_distribution<double> norm(0.0,1.0);
