.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -pg  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c -pg $*.cpp

firstGA = FirstRealGA FirstBitGA

ALL = $(firstGA) exercise1.3

lesson1 : $(firstGA)

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~
