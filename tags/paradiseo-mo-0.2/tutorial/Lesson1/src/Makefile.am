INCLUDES = -I${EO_DIR}/src/ -I ../../../src/ -I ../../examples/tsp/src/

AM_CXXFLAGS =\
	-Wall\
	-Werror\
	-Wno-deprecated\
	-ansi\
	-pedantic

bin_PROGRAMS = hill_climbing

##############################
hill_climbing_SOURCES = hill_climbing.cpp

hill_climbing_LDFLAGS = 

hill_climbing_LDADD = \
	${EO_DIR}/src/libeo.a\
	${EO_DIR}/src/utils/libeoutils.a\
	../../examples/tsp/src/libtsp.a 
