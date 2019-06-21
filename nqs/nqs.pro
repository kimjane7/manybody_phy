TEMPLATE = app
CONFIG += console c++11
CONFIG += object_parallel_to_source
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += src/main.cpp \
           src/neuralquantumstate/neuralquantumstate.cpp \
           src/hamiltonian/hamiltonian.cpp \
           src/optimizer/optimizer.cpp \
           src/optimizer/sgd/sgd.cpp \
           src/sampler/sampler.cpp \
           src/sampler/metropolis/metropolis.cpp \
           src/sampler/metropolis/bruteforce/bruteforce.cpp \
           src/sampler/metropolis/importancesampling/importancesampling.cpp
           
HEADERS += src/main.h \
           src/neuralquantumstate/neuralquantumstate.h \
           src/hamiltonian/hamiltonian.h \
           src/optimizer/optimizer.h \
           src/optimizer/sgd/sgd.h \
           src/sampler/sampler.h \
           src/sampler/metropolis/metropolis.h \
           src/sampler/metropolis/bruteforce/bruteforce.h \
           src/sampler/metropolis/importancesampling/importancesampling.h
