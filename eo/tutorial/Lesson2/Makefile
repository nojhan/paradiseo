.cpp: ; g++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; g++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c $*.cpp

firstEA = FirstRealEA FirstBitEA

ALL = $(firstEA) exercise1 exercise2 exercise3 

lesson2 : $(firstEA)

clean : 
	@/bin/rm $(ALL) *.o *~
