# Process this file with automake to produce Makefile.in

CXX=g++

INCLUDES = \
	-I$(EO_DIR)/src \
	-I$(MO_DIR)/src \
	-I$(MOEO_DIR)/src 

AM_CXXFLAGS =\
	-Wall\
	-g

SUBDIRS= rmc

lib_LIBRARIES = libparadiseo.a

#bin_PROGRAMS = peodeploy essai

## Shared sources 

shared_SOURCES = \
	paradiseo \
	paradiseo.h \
	core/peo_init.h \
	core/peo_init.cpp \
	core/peo_fin.h \
	core/peo_fin.cpp \
	core/peo_run.h \
	core/peo_run.cpp \
	core/peo_param.h \
	core/peo_param.cpp \
	core/peo_debug.h \
	core/peo_debug.cpp \
	core/thread.h \
	core/thread.cpp \
	core/reac_thread.h \
	core/reac_thread.cpp \
	core/service.h \
	core/service.cpp \
	core/runner.h \
	core/runner.cpp \
	core/communicable.h \
	core/communicable.cpp \
	core/cooperative.h \
	core/rmc.h \
	core/messaging.h \
	core/topology.h \
	core/topology.cpp \
	core/ring_topo.h \
	core/ring_topo.cpp \
	core/eoPop_comm.h \
	peoEA.h \
	peoTransform.h \
	peoSeqTransform.h \
	peoParaTransform.h \
	peoPopEval.h\
	peoSeqPopEval.h \
	peoAggEvalFunc.h \
	peoParaPopEval.h

## Main library

libparadiseo_a_SOURCES = \
	$(shared_SOURCES)

#libparadiseo_a_LIBADD = 

## Miscellaneous Tools

# peodeploy