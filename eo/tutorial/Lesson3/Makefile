.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c $*.cpp

secondEA = SecondBitEA

ALL = $(secondEA) exercise3.1

lesson3 : $(secondEA)

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *.sav *.xg *.status *~
