# sample makefile for building an EA evolving a new genotype

.cpp: ; c++  -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -Wall -g  -o $@  $*.cpp ../../src/libeo.a ../../src/utils/libeoutils.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -DVERSION=\"0.9.3\" -I. -I../../src -I./util -Wall -g -c $*.cpp

# local sources
COMMON_SOURCES =    eoOneMax.h \
   eoOneMaxEvalFunc.h \
   eoOneMaxInit.h \
   eoOneMaxMutation.h \
   eoOneMaxQuadCrossover.h \
   make_genotype_OneMax.h \
   make_op_OneMax.h

NO_LIB_SOURCES = OneMaxEA.cpp

LIB_SOURCES = OneMaxLibEA.cpp make_OneMax.cpp


SOURCES = $(COMMON_SOURCES) OneMaxEA.cpp OneMaxLibEA.cpp make_OneMax.cpp

# START eventually modify the name of EO dir
DIR_EO = ../../src
# END eventually modify the name of EO dir

LIB_EO = $(DIR_EO)/utils/libeoutils.a $(DIR_EO)/libeo.a

ALL = OneMaxEA OneMaxLibEA

OneMaxEA : OneMaxEA.o
	c++ -g -o $@ OneMaxEA.o ../../src/utils/libeoutils.a ../../src/libeo.a -lm 

OneMaxLibEA : OneMaxLibEA.o make_OneMax.o
	c++ -g -o $@ OneMaxLibEA.o make_OneMax.o ../../src/utils/libeoutils.a ../../src/libeo.a -lm

tar : ; tar czvf OneMax.tgz *.h *.cpp Makefile

all : $(ALL)

clean : ; /bin/rm  *.o $(ALL)

########## local dependencies
OneMaxEA.o : $(COMMON_SOURCES) OneMaxEA.cpp
OneMaxLibEA.o : $(COMMON_SOURCES) OneMaxLibEA.cpp
make_OneMax.o : make_OneMax.cpp eoOneMax.h 
