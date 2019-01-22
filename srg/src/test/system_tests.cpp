#include "catch.hpp"
#include "../system.h"
#include <random>

TEST_CASE("STATES"){

	CFermionSystem PairingModel(4, 4, 1.0, 0.5);

	REQUIRE(PairingModel.states_(0,0) == 0);
	REQUIRE(PairingModel.states_(0,1) == 1);

	REQUIRE(PairingModel.states_(1,0) == 0);
	REQUIRE(PairingModel.states_(1,1) == -1);

	REQUIRE(PairingModel.states_(2,0) == 1);
	REQUIRE(PairingModel.states_(2,1) == 1);

	REQUIRE(PairingModel.states_(3,0) == 1);
	REQUIRE(PairingModel.states_(3,1) == -1);

	REQUIRE(PairingModel.states_(4,0) == 2);
	REQUIRE(PairingModel.states_(4,1) == 1);

	REQUIRE(PairingModel.states_(5,0) == 2);
	REQUIRE(PairingModel.states_(5,1) == -1);

	REQUIRE(PairingModel.states_(6,0) == 3);
	REQUIRE(PairingModel.states_(6,1) == 1);

	REQUIRE(PairingModel.states_(7,0) == 3);
	REQUIRE(PairingModel.states_(7,1) == -1);
}

TEST_CASE("KINETIC ENERGY"){

	CFermionSystem PairingModel(4, 4, 1.0, 0.5);

	REQUIRE(PairingModel.h0(0,0) == 0);
	REQUIRE(PairingModel.h0(1,1) == 0);
	REQUIRE(PairingModel.h0(2,2) == 1);
	REQUIRE(PairingModel.h0(3,3) == 1);
	REQUIRE(PairingModel.h0(4,4) == 2);
	REQUIRE(PairingModel.h0(5,5) == 2);
	REQUIRE(PairingModel.h0(6,6) == 3);
	REQUIRE(PairingModel.h0(7,7) == 3);
}

TEST_CASE("FOCK ENERGY"){

	mt19937 generator;
	uniform_real_distribution<double> rand01(0.0,1.0);
	double g = rand01(generator);

	CFermionSystem PairingModel(4, 4, 1.0, g);

	REQUIRE(PairingModel.f(0,0) == -0.5*g);
	REQUIRE(PairingModel.f(1,1) == -0.5*g);
	REQUIRE(PairingModel.f(2,2) == 1.0-0.5*g);
	REQUIRE(PairingModel.f(3,3) == 1.0-0.5*g);
	REQUIRE(PairingModel.f(4,4) == 2.0);
	REQUIRE(PairingModel.f(5,5) == 2.0);
	REQUIRE(PairingModel.f(6,6) == 3.0);
	REQUIRE(PairingModel.f(7,7) == 3.0);
}