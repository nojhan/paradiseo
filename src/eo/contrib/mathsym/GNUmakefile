COMPILEFLAGS=-Wno-deprecated -g -Wall -mpreferred-stack-boundary=2 -falign-functions=0#-DINTERVAL_DEBUG
OPTFLAGS= #-O3 -DNDEBUG 

PROFILE_FLAGS=#-pg 
LDFLAGS=#-a

INCLUDES=-I. -Isym -Ifun -Igen -Ieval -Iregression -I../../src -Ieo_interface  -I..

CPPFLAGS=$(COMPILEFLAGS) $(OPTFLAGS) $(INCLUDES) $(PROFILE_FLAGS) -D__I386__ -DSIZEOF_UNSIGNED_LONG=4
EXTLIBS=tcc/libtcc.a tcc/libtcc1.a ../../src/libeo.a ../../src/utils/libeoutils.a  

LIBS=${EXTLIBS} -ldl 

SYMLIB=libsym.a

VPATH=sym fun gen eval regression eo_interface

CXXSOURCES=FunDef.cpp Sym.cpp SymImpl.cpp SymOps.cpp sym_compile.cpp TreeBuilder.cpp LanguageTable.cpp\
	Dataset.cpp ErrorMeasure.cpp Scaling.cpp TargetInfo.cpp BoundsCheck.cpp util.cpp NodeSelector.cpp\
	eoSymCrossover.cpp sym_operations.cpp eoSymMutate.cpp eoSymLambdaMutate.cpp MultiFunction.cpp

TESTPROGRAMS=test/test_compile test/testeo test/test_simplify test/test_diff test/test_lambda test/test_mf test/test_interval

OBJS= $(CXXSOURCES:.cpp=.o) c_compile.o

all: tcc/ symreg

include $(CXXSOURCES:.cpp=.d) symreg.d 

clean:
	rm *.o *.d $(TESTPROGRAMS) $(SYMLIB) symreg test/*.o || true

distclean: clean
	rm -rf tcc

symreg: libsym.a symreg.o $(EXTLIBS)
	$(CXX) -o symreg symreg.o libsym.a $(LIBS) $(PROFILE_FLAGS) ${LDFLAGS}

libsym.a: $(OBJS) 
	rm libsym.a; ar cq $(SYMLIB) $(OBJS) 

check: $(TESTPROGRAMS)
	test/test_compile && test/test_interval && test/testeo && test/test_simplify && test/test_diff && test/test_lambda && echo "all tests succeeded"

test/test_compile: test/test_compile.o ${SYMLIB}
	$(CXX) -o test/test_compile test/test_compile.o $(SYMLIB) ${LIBS}

test/testeo: test/testeo.o ${SYMLIB}
	$(CXX) -o test/testeo test/testeo.o  $(SYMLIB) ${LIBS}

test/test_simplify: test/test_simplify.o $(SYMLIB)
	$(CXX) -o test/test_simplify test/test_simplify.o  $(SYMLIB) ${LIBS}

test/test_diff: test/test_diff.o $(SYMLIB)
	$(CXX) -o test/test_diff test/test_diff.o  $(SYMLIB) ${LIBS}

test/test_lambda: test/test_lambda.o $(SYMLIB)
	$(CXX) -o test/test_lambda test/test_lambda.o  $(SYMLIB) ${LIBS}

test/test_mf: test/test_mf.o $(SYMLIB)
	$(CXX) -o test/test_mf test/test_mf.o  $(SYMLIB) ${LIBS}

test/test_interval: test/test_interval.o
	$(CXX) -o test/test_interval test/test_interval.o  $(SYMLIB) ${LIBS}


# eo
../../src/libeo.a:
	make -C ../../src libeo.a

../../src/utils/libeoutils.a:
	make -C ../../src/utils libeoutils.a

# tiny cc
tcc/: tcc.tar.gz
	tar xvfz tcc.tar.gz && cd tcc && ./configure && make

tcc/Makefile: tcc/
	cd tcc && ./configure

tcc/libtcc.a: tcc/Makefile
	make -Ctcc

tcc/libtcc1.a: tcc/Makefile
	make -Ctcc

#rules
c_compile.o: eval/c_compile.c
	$(CC) -c eval/c_compile.c -I./tcc $(COMPILEFLAGS) $(OPTFLAGS)

%.o:%.cpp
	$(CXX) -o $@ -c $< $(CPPFLAGS) $(INCLUDE)

%.d: %.cpp
	$(SHELL) -ec '$(CXX) -M $(CPPFLAGS) $< | sed "s/$*.o/& $@/g" > $@ '


%.d: %.c
	$(SHELL) -ec '$(CXX) -M $(CPPFLAGS) $< | sed "s/$*.o/& $@/g" > $@ '


