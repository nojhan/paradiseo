# Note for however is foolish enough to attempt to build this thing
#
# You need: 
# Python 2.2
# Boost.Python v2
#

CXX = g++ #-3.2
CXXFLAGS = -DHAVE_SSTREAM#-g #-DNDEBUG
CPPFLAGS = -Wall -O2 #-g #-O2
LDFLAGS = 
COMPILE = $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c
LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) 
INC=-I/usr/include/python2.2 -I/home/mkeijzer/src/boost -I.. -ftemplate-depth-50 #-I/usr/include/stlport

OBJECTS=eoFunctorStore.o PyEO.o abstract1.o algos.o \
	random_numbers.o geneticOps.o selectOne.o continuators.o\
	reduce.o replacement.o selectors.o breeders.o\
	mergers.o valueParam.o perf2worth.o monitors.o\
	statistics.o

all: PyEO/PyEO.so

clean:
	rm PyEO/*.so *.o test/*.pyc

PyEO/PyEO.so: $(OBJECTS)
	$(LINK) -o PyEO/PyEO.so $(OBJECTS) -lboost_python -lpython2.2 -shared #-lstlport

eoFunctorStore.o: ../eoFunctorStore.h ../eoFunctorStore.cpp
	$(COMPILE) -o eoFunctorStore.o ../eoFunctorStore.cpp $(INC)

%.o:%.cpp PyEO.h def_abstract_functor.h 
	$(COMPILE) $< $(INC)


