## Makefile.am for eo/src/utils

lib_LIBRARIES 		= libeoutils.a

libeoutils_a_SOURCES 	= eoData.cpp \
			eoFileMonitor.cpp \
			eoGnuplot.cpp \
			eoGnuplot1DMonitor.cpp \
			eoGnuplot1DSnapshot.cpp \
			eoIntBounds.cpp \
			eoParser.cpp \
			eoRealBounds.cpp \
			eoRNG.cpp \
			eoState.cpp \
			eoStdoutMonitor.cpp \
			eoUpdater.cpp \
			make_help.cpp \
			pipecom.cpp

utilsincludedir		= $(pkgincludedir)/utils

utilsinclude_HEADERS 	= checkpointing \
			compatibility.h \
			eoAssembledFitnessStat.h \
			eoCheckPoint.h \
			eoData.h \
			eoDistance.h \
			eoFDCStat.h \
			eoFileMonitor.h \
			eoFileSnapshot.h \
			eoGnuplot.h \
			eoGnuplot1DMonitor.h \
			eoGnuplot1DSnapshot.h \
			eoHowMany.h \
			eoMOFitnessStat.h \
			eoMonitor.h \
			eoParam.h \
			eoParser.h \
			eoPopStat.h \
			eoRNG.h \
			eoRealBounds.h \
			eoIntBounds.h \
			eoRealVectorBounds.h \
			eoRndGenerators.h \
			eoScalarFitnessStat.h \
			eoStat.h \
			eoState.h \
			eoStdoutMonitor.h \
			eoTimeCounter.h \
			eoTimedMonitor.h \
			eoUniformInit.h \
			eoUpdatable.h \
			eoUpdater.h \
			pipecom.h \
			rnd_generators.h \
			selectors.h

AM_CXXFLAGS 		= -I$(top_srcdir)/src -DGNUPLOT_PROGRAM=\"@GNUPLOT@\"
