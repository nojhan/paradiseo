# Note for however is foolish enough to attempt to build this thing
#
# You need: 
# Python 2.2
# Stlport
# Boost.Python v2
#
#
# On my debian (unstable), I used libstlport_gcc.so.4.5
# and libboost_python.so.1.29.0
#
# Obviously together with python2.2 (as Boost.Python.v2 needs that)
#
# I'm pretty sure any stlport will do, but less convinced about boost.python
# That lib seems to be pretty much under development (as I found out, the hard way)
# but this version 1.29 seems to work for me.
#
# My version of boost was found in /usr/include, modify INC to your system
# 
# For you happy Debian unstable users, just install
#
# apt-get install libstlport4.5
# apt-get install libboost-python-dev
# apt-get install libboost-python1.29.0
#

CXX = g++ #-3.2
CXXFLAGS = -DHAVE_SSTREAM#-g #-DNDEBUG
CPPFLAGS = -Wall -O2 #-g #-O2
LDFLAGS = 
COMPILE = $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c
LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 
INC=-I/usr/include/python2.2 -I.. -ftemplate-depth-50 -I/usr/include/stlport

OBJECTS=eoFunctorStore.o PyEO.o abstract1.o algos.o \
	random_numbers.o geneticOps.o selectOne.o continuators.o\
	reduce.o replacement.o selectors.o breeders.o\
	mergers.o valueParam.o perf2worth.o monitors.o\
	statistics.o

all: PyEO/PyEO.so

clean:
	rm PyEO/*.so *.o test/*.pyc

PyEO/PyEO.so: $(OBJECTS)
	$(LINK) -o PyEO/PyEO.so $(OBJECTS) -lboost_python -lpython2.2 -shared -lstlport

eoFunctorStore.o: ../eoFunctorStore.h ../eoFunctorStore.cpp
	$(COMPILE) -o eoFunctorStore.o ../eoFunctorStore.cpp $(INC)

%.o:%.cpp PyEO.h def_abstract_functor.h 
	$(COMPILE) $< $(INC)


