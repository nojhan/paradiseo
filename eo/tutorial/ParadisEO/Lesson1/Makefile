MPICC = ./mpiCC

ALL = IslandBitEA IslandBitEA2 IslandBitEA1

lesson2 : IslandBitEA IslandBitEA2 IslandBitEA1

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~

IslandBitEA : IslandBitEA.o
	$(MPICC) -DPACKAGE=\"eo\" -o IslandBitEA IslandBitEA.o ../../../src/utils/libeoutils.a ../../../src/libeo.a

IslandBitEA.o : IslandBitEA.cpp binary_value.h
	$(MPICC) -DPACKAGE=\"eo\" -I. -I../../../src -c IslandBitEA.cpp

IslandBitEA1 : IslandBitEA1.o
	$(MPICC) -DPACKAGE=\"eo\" -o IslandBitEA1 IslandBitEA1.o ../../../src/utils/libeoutils.a ../../../src/libeo.a

IslandBitEA1.o : IslandBitEA1.cpp binary_value.h
	$(MPICC) -DPACKAGE=\"eo\" -I. -I../../../src -c IslandBitEA1.cpp

IslandBitEA2 : IslandBitEA2.o
	$(MPICC) -DPACKAGE=\"eo\" -o IslandBitEA2 IslandBitEA2.o ../../../src/utils/libeoutils.a ../../../src/libeo.a

IslandBitEA2.o : IslandBitEA2.cpp binary_value.h
	$(MPICC) -DPACKAGE=\"eo\" -I. -I../../../src -c IslandBitEA2.cpp

