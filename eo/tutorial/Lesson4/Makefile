.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/es/libes.a ../../src/utils/libeoutils.a ../../src/libeo.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.1\" -I. -I../../src -Wall -g -c $*.cpp


ALL = BitEA RealEA ESEA

lesson3 : $(ALL)

all : $(ALL)

BitEA : BitEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/ga/libga.a ../../src/utils/libeoutils.a ../../src/libeo.a

RealEA : RealEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/es/libes.a ../../src/utils/libeoutils.a ../../src/libeo.a

ESEA : ESEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/es/libes.a ../../src/utils/libeoutils.a ../../src/libeo.a

clean : 
	@/bin/rm $(ALL) *.o *.sav *.xg *.status *~
