ALL = MasterDistEvalBitEA SlaveDistEvalBitEA 

lesson2 : MasterDistEvalBitEA SlaveDistEvalBitEA 

all : $(ALL)

clean : 
	@/bin/rm $(ALL) *.o *~

MasterDistEvalBitEA : MasterDistEvalBitEA.o
	mpiCC -DPACKAGE=\"eo\" -o MasterDistEvalBitEA MasterDistEvalBitEA.o ../../../src/utils/libeoutils.a ../../../src/libeo.a

MasterDistEvalBitEA.o : MasterDistEvalBitEA.cpp binary_value.h
	mpiCC -DPACKAGE=\"eo\" -I. -I../../../src -c MasterDistEvalBitEA.cpp

SlaveDistEvalBitEA : SlaveDistEvalBitEA.o
	mpiCC -DPACKAGE=\"eo\" -o SlaveDistEvalBitEA SlaveDistEvalBitEA.o ../../../src/utils/libeoutils.a ../../../src/libeo.a

SlaveDistEvalBitEA.o : SlaveDistEvalBitEA.cpp binary_value.h
	mpiCC -DPACKAGE=\"eo\" -I. -I../../../src -c SlaveDistEvalBitEA.cpp

