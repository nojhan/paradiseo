.cpp: ; g++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; g++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c $*.cpp

firstGA = FirstRealGA FirstBitGA

ALL = $(firstGA) exercise3

lesson1 : $(firstGA)

clean : 
	@/bin/rm $(ALL) *.o *~
