### This Makefile is part of the tutorial of the EO library
# Unlike other Makefiles in EO, it is not using the automake/autoconf 
# so that it stays easy to understant (you are in the tutorial, remember!)
# MS, Oct. 2002

# if you use this Makefile as a starting point for another application
# you might need to modify the following
DIR_EO = ../../src

.SUFFIXES: .cpp

# Warning: $(CXX) in Linux (RedHat and Mandrake at least) is g++
# However, if you are using this Makefile within xemacs, 
# and have problems with the interpretation of the output (and its colors)
# then you should use c++ instead (make CXX=c++ will do)

.cpp: ; $(CXX)  -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -Wall -g -pg  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; $(CXX) -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -Wall -g -c -pg $*.cpp

ALL = BitEA RealEA ESEA

lesson3 : $(ALL)

all : $(ALL)

BitEA : BitEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/ga/libga.a ../../src/utils/libeoutils.a ../../src/libeo.a

RealEA : RealEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/es/libes.a ../../src/utils/libeoutils.a ../../src/libeo.a

ESEA : ESEA.o ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.2\" -Wall -g  -o $@  $< ../../src/es/libes.a ../../src/utils/libeoutils.a ../../src/libeo.a

clean : 
	@/bin/rm $(ALL) *.o *.sav *.xg *.status *~
