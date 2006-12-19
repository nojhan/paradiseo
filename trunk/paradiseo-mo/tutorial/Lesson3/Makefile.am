INCLUDES = -I${EO_DIR}/src/ -I ../../src/ -I ../examples/tsp/src/

AM_CXXFLAGS =\
	-Wall\
	-Werror\
	-Wno-deprecated\
	-ansi\
	-pedantic

bin_PROGRAMS = simulated_annealing

##############################
simulated_annealing_SOURCES = simulated_annealing.cpp

simulated_annealing_LDFLAGS = 

simulated_annealing_LDADD = \
	${EO_DIR}/src/libeo.a\
	${EO_DIR}/src/utils/libeoutils.a\
	../examples/tsp/src/libtsp.a 
