CXX=mpicxx

INCLUDES = \
	 -I$(EO_DIR)/src/ \
	 -I$(MO_DIR)/src/ \
	 -I../../src/ 

AM_CXXFLAGS = \
	 -Wall -g

bin_PROGRAMS = \
	tspExample 

tspExample_SOURCES = \
	main.cpp 

tspExample_LDFLAGS = \
	-L$(EO_DIR)/src/ \
	-L$(EO_DIR)/src/utils \
	-L../../src/ \
	-L../../src/rmc/mpi/ \
	`xml2-config --libs` 

tspExample_LDADD = -lparadiseo -lrmc_mpi -leoutils -leo -ltsp_shared

