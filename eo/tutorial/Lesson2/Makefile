.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c $*.cpp

firstEA = FirstRealEA FirstBitEA

ALL = $(firstEA) exercise2.3

lesson2 : $(firstEA)

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~

FirstRealEA : real_value.h

FirstBitEA : binary_value.h