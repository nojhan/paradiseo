.SUFFIXES: .cpp
.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../../src -Wall -g  -o $@  $*.cpp ../../../src/utils/libeoutils.a ../../../src/libeo.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../../src -Wall -g -c $*.cpp

ALL = CellularBitEA

lesson3 : $(firstEA)

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~

CellularBitEA : binary_value.h

