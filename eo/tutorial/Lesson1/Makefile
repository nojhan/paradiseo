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

firstGA = FirstRealGA FirstBitGA

ALL = $(firstGA) exercise1.3

lesson1 : $(firstGA)

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~
