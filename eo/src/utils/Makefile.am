###############################################################################
##
## Makefile.am for eo/src/utils
##
###############################################################################

INCLUDES = -I$(top_builddir)/src
CPPFLAGS = -O2 -Wall
lib_LIBRARIES = libeoutils.a
libeoutils_a_SOURCES = eoParser.cpp eoRNG.cpp eoState.cpp eoUpdater.cpp eoFileMonitor.cpp eoStdoutMonitor.cpp eoRealBounds.cpp make_help.cpp

libeoincdir = $(includedir)/eo/utils
libeoinc_HEADERS = checkpointing *.h
#compatibility.h eoCheckPoint.h eoData.h eoDistance.h eoFDCStat.h eoFileMonitor.h eoGnuplot1DMonitor.h eoGnuplot1DSnapshot.h eoHowMany.h eoMonitor.h eoParam.h eoParser.h eoRNG.h eoStat.h eoScalarFitnessStat.h eoState.h eoStdoutMonitor.h eoUpdatable.h eoUpdater.h rnd_generators.h eoRndGenerators.h selectors.h



