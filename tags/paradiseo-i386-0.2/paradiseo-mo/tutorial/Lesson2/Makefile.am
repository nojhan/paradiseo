INCLUDES = -I${EO_DIR}/src/ -I ../../src/ -I ../examples/tsp/src/

AM_CXXFLAGS =\
	-Wall\
	-Werror\
	-Wno-deprecated\
	-ansi\
	-pedantic

bin_PROGRAMS = tabu_search

##############################
tabu_search_SOURCES = tabu_search.cpp

tabu_search_LDFLAGS = 

tabu_search_LDADD = \
	${EO_DIR}/src/libeo.a\
	${EO_DIR}/src/utils/libeoutils.a\
	../examples/tsp/src/libtsp.a 
