###############################################################################
##
## Makefile.am for eo/test
##
###############################################################################

LIBEO				= $(top_builddir)/src/libeo.a
LIBEOUTILS			= $(top_builddir)/src/utils/libeoutils.a
LIBES				= $(top_builddir)/src/es/libes.a
LIBGA				= $(top_builddir)/src/ga/libga.a

# PLEASE don't break the line (see create_batch.sh)
check_PROGRAMS 			= t-eoParetoFitness \
				t-eoPareto \
				t-eofitness \
				t-eoRandom \
				t-eobin \
				t-eoVirus \
				t-MGE \
				t-MGE1bit \
				t-MGE-control \
				t-eoStateAndParser \
				t-eoCheckpointing \
				t-eoSSGA \
				t-eoExternalEO \
				t-eoSymreg \
				t-eo \
				t-eoReplacement \
				t-eoSelect \
				t-eoGenOp \
				t-eoGA \
				t-eoReal \
				t-eoVector \
				t-eoESAll \
				t-eoPBIL \
				t-eoFitnessAssembled \
				t-eoFitnessAssembledEA \
				t-eoRoulette \
				t-eoSharing


TESTS				= $(check_PROGRAMS) \
				run_tests # This script can be used to check command-line arguments

noinst_HEADERS			= binary_value.h real_value.h RoyalRoad.h


AM_CXXFLAGS			= -I$(top_srcdir)/src 
DEPS	 			= $(LIBGA) $(LIBES) $(LIBEOUTILS) $(LIBEO)
LIBS	 			= $(DEPS)

CLEANFILES 			= monitor.csv t-eoRandom.out
EXTRA_DIST			= run_tests

# extra flags for specific targets
t_eoVirus_CXXFLAGS		= $(AM_CXXFLAGS) -I$(top_srcdir)/contrib

# Specify source-files,
# otherwise automake/make looks for C sources
#
t_eoRandom_SOURCES 		= t-eoRandom.cpp
t_eofitness_SOURCES 		= t-eofitness.cpp
t_eoFitnessAssembledEA_SOURCES 	= t-eoFitnessAssembledEA.cpp
t_eobin_SOURCES 		= t-eobin.cpp
t_eoVirus_SOURCES 		= t-eoVirus.cpp
t_MGE1bit_SOURCES 		= t-MGE1bit.cpp
t_MGE_SOURCES 			= t-MGE.cpp
t_MGE_control_SOURCES 		= t-MGE-control.cpp
t_eoStateAndParser_SOURCES 	= t-eoStateAndParser.cpp
t_eoCheckpointing_SOURCES 	= t-eoCheckpointing.cpp
t_eoReplacement_SOURCES 	= t-eoReplacement.cpp
t_eoSelect_SOURCES 		= t-eoSelect.cpp
t_eoExternalEO_SOURCES 		= t-eoExternalEO.cpp
t_eoSymreg_SOURCES 		= t-eoSymreg.cpp
t_eo_SOURCES 			= t-eo.cpp
t_eoGenOp_SOURCES 		= t-eoGenOp.cpp
t_eoVector_SOURCES 		= t-eoVector.cpp
t_eoGA_SOURCES 			= t-eoGA.cpp
t_eoReal_SOURCES 		= t-eoReal.cpp
t_eoESAll_SOURCES 		= t-eoESAll.cpp
t_eoSSGA_SOURCES 		= t-eoSSGA.cpp
t_eoPareto_SOURCES 		= t-eoPareto.cpp 
t_eoParetoFitness_SOURCES 	= t-eoParetoFitness.cpp 
t_eoPBIL_SOURCES 		= t-eoPBIL.cpp 
t_eoFitnessAssembled_SOURCES 	= t-eoFitnessAssembled.cpp 
t_eoRoulette_SOURCES 		= t-eoRoulette.cpp
t_eoSharing_SOURCES 		= t-eoSharing.cpp
