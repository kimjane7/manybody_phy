CC = g++
CFLAGS = -std=c++11
LIB = -larmadillo

all: run 



run: main.o RBM.o
	${CC} ${CFLAGS} main.o RBM.o -o run ${LIB}



main.o: main.cpp 
	${CC} ${CFLAGS} -c main.cpp
boson_system.o: boson_system.cpp
	${CC} ${CFLAGS} -c boson_system.cpp ${LIB}
RBM.o: RBM.cpp
	${CC} ${CFLAGS} -c RBM.cpp ${LIB}



clean: 
	rm run
	rm *o *.dat

