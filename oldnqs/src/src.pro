TEMPLATE = lib
CONFIG += staticlib
CONFIG += object_parallel_to_source
CONFIG += console c++11
CONFIG -= qt

SOURCES += neuralquantumstate/neuralquantumstate.cpp \
           hamiltonian/hamiltonian.cpp \
           optimizer/optimizer.cpp \
           optimizer/sgd/sgd.cpp \
           sampler/sampler.cpp \
           sampler/metropolis/metropolis.cpp \
           sampler/metropolis/bruteforce/bruteforce.cpp \
           sampler/metropolis/importancesampling/importancesampling.cpp
           
HEADERS += neuralquantumstate/neuralquantumstate.h \
           hamiltonian/hamiltonian.h \
           optimizer/optimizer.h \
           optimizer/sgd/sgd.h \
           sampler/sampler.h \
           sampler/metropolis/metropolis.h \
           sampler/metropolis/bruteforce/bruteforce.h \
           sampler/metropolis/importancesampling/importancesampling.h