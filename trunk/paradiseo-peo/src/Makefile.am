# Process this file with automake to produce Makefile.in

CXX=g++

INCLUDES = \
	-I$(EO_DIR)/src 

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
	peo_init.h \
	peo_init.cpp \
	peo_fin.h \
	peo_fin.cpp \
	peo_run.h \
	peo_run.cpp \
	peo_param.h \
	peo_param.cpp \
	peo_debug.h \
	peo_debug.cpp \
	thread.h \
	thread.cpp \
	reac_thread.h \
	reac_thread.cpp \
	service.h \
	service.cpp \
	runner.h \
	runner.cpp \
	comm.h \
	comm.cpp \
	coop.h \
	rmc.h \
	comm.h \
	mess.h \
	topo.h \
	topo.cpp \
	ring_topo.h \
	ring_topo.cpp \
	eoPop_comm.h \
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



