CXX = g++
CXXFLAGS = -std=c++11




all: run

run: main.o neuralquantumstate.o hamiltonian.o optimizer.o sgd.o sampler.o metropolis.o bruteforce.o importancesampling.o
	$(CXX) $(CXXFLAGS) main.o neuralquantumstate.o hamiltonian.o optimizer.o sgd.o sampler.o metropolis.o bruteforce.o importancesampling.o -o run




main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp

neuralquantumstate.o: src/neuralquantumstate/neuralquantumstate.cpp
	$(CXX) $(CXXFLAGS) -c src/neuralquantumstate/neuralquantumstate.cpp

hamiltonian.o: src/hamiltonian/hamiltonian.cpp
	$(CXX) $(CXXFLAGS) -c src/hamiltonian/hamiltonian.cpp

optimizer.o: src/optimizer/optimizer.cpp
	$(CXX) $(CXXFLAGS) -c src/optimizer/optimizer.cpp

sgd.o: src/optimizer/sgd/sgd.cpp
	$(CXX) $(CXXFLAGS) -c src/optimizer/sgd/sgd.cpp

sampler.o: src/sampler/sampler.cpp
	$(CXX) $(CXXFLAGS) -c src/sampler/sampler.cpp

metropolis.o: src/sampler/metropolis/metropolis.cpp
	$(CXX) $(CXXFLAGS) -c src/sampler/metropolis/metropolis.cpp

bruteforce.o: src/sampler/metropolis/bruteforce/bruteforce.cpp
	$(CXX) $(CXXFLAGS) -c src/sampler/metropolis/bruteforce/bruteforce.cpp

importancesampling.o: src/sampler/metropolis/importancesampling/importancesampling.cpp
	$(CXX) $(CXXFLAGS) -c src/sampler/metropolis/importancesampling/importancesampling.cpp



clean:
	rm *.o 
	rm *.dat
	rm run

