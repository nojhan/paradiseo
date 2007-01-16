# Process this file with automake to produce Makefile.in

CXX=g++

INCLUDES = \
	-I$(EO_DIR)/src \
	-I$(MO_DIR)/src \
	-I$(MOEO_DIR)/src 

AM_CXXFLAGS =\
	-Wall\
	-g

lib_LIBRARIES = libparadiseo.a

#bin_PROGRAMS = peodeploy essai

## Shared sources 

shared_SOURCES = \
	peo_init.cpp \
	peo_fin.cpp \
	peo_run.cpp \
	peo_param.cpp \
	peo_debug.cpp \
	thread.cpp \
	reac_thread.cpp \
	service.cpp \
	runner.cpp \
	communicable.cpp \
	topology.cpp \
	ring_topo.cpp


## Main library

libparadiseo_a_SOURCES = \
	$(shared_SOURCES)

#libparadiseo_a_LIBADD = 

## Miscellaneous Tools

# peodeploy
