TEMPLATE = app
TARGET = test
CONFIG += console c++11

SOURCES += main.cpp tests.cpp
HEADERS += catch.hpp
INCLUDEPATH += ../src/
LIBS += -L../src -lsrc
PRE_TARGETDEPS += ../src/libsrc.a