#include "catch.hpp"
#include "../imsrg.h"

TEST_CASE("ONE-BODY COMPONENT OF NORMAL-ORDERED HAMILTONIAN"){

	double d = 1.0;
	double g = 0.55;
	double smax = 10.0;
	double ds = 0.001;

	ivec holes = {0,1,2,3};
	ivec parts = {4,5,6,7};
	CIMSRG PairingModel(8, d, g, smax, ds, holes, parts);

	REQUIRE(PairingModel.f_(0,0) == -0.5*g);
	REQUIRE(PairingModel.f_(1,1) == -0.5*g);
	REQUIRE(PairingModel.f_(2,2) == d-0.5*g);
	REQUIRE(PairingModel.f_(3,3) == d-0.5*g);
	REQUIRE(PairingModel.f_(4,4) == 2.0*d);
	REQUIRE(PairingModel.f_(5,5) == 2.0*d);
	REQUIRE(PairingModel.f_(6,6) == 3.0*d);
	REQUIRE(PairingModel.f_(7,7) == 3.0*d);
}
