CC = g++
CFLAGS = -std=c++11 -Wall
LIB = -larmadillo


all: IMSRG SRG

MBPT2: mbpt2.o solver.o system.o
	${CC} ${CFLAGS} mbpt2.o system_tests.o system.o -o MBPT2

TEST: test_main.o system_tests.o system.o
	${CC} ${CFLAGS} test_main.o system_tests.o system.o -o TEST

SRG: srg.o
	${CC} ${CFLAGS} srg.o -o SRG ${LIB}

IMSRG: imsrg.o
	${CC} ${CFLAGS} imsrg.o -o IMSRG ${LIB}




mbpt2.o: mbpt2.cpp 
	${CC} ${CFLAGS} -c mbpt2.cpp

srg.o: srg.cpp
	${CC} ${CFLAGS} -c srg.cpp ${LIB}

imsrg.o: imsrg.cpp
	${CC} ${CFLAGS} -c imsrg.cpp ${LIB}

solver.o: solver.cpp
	${CC} ${CFLAGS} -c solver.cpp

system.o: system.cpp
	${CC} ${CFLAGS} -c system.cpp

test_main.o: test/test_main.cpp
	${CC} ${CFLAGS} -c test/test_main.cpp ${LIB}

system_tests.o: test/system_tests.cpp
	${CC} ${CFLAGS} -c test/system_tests.cpp

imsrg_tests.o: test/imsrg_tests.cpp
	${CC} ${CFLAGS} -c test/imsrg_tests.cpp



clean: 
	rm *.o
	rm IMSRG
	rm SRG
	rm TEST

