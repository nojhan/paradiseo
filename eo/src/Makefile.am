###############################################################################
##
## Makefile.am for eo/src
##
###############################################################################

SUBDIRS 	= es ga gp utils other do

AM_CXXFLAGS	= -I$(top_srcdir)/src

lib_LIBRARIES = libeo.a

libeo_a_SOURCES = eoFunctorStore.cpp \
		eoPersistent.cpp
		eoPrintable.cpp \
		eoCtrlCContinue.cpp \
		eoParetoFitness.cpp \
		eoScalarFitnessAssembled.cpp  

libeoincdir = $(includedir)/eo

libeoinc_HEADERS = $(srcdir)/*.h $(srcdir)/eo $(srcdir)/do/*.h 

