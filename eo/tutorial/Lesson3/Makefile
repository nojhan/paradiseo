.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/utils/libeoutils.a ../../src/libeo.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -Wall -g -c $*.cpp

secondEA = SecondBitEA SecondRealEA

ALL = $(secondEA) exercise3.1

lesson3 : $(secondEA)

all : $(ALL)

SecondBitEA : binary_value.h
SecondRealEA : real_value.h

clean : 
	@/bin/rm $(ALL) *.o *.sav *.xg *.status *~
