###############################################################################
##
## Makefile.am for eo/src
##
###############################################################################

SUBDIRS 		= es ga gp utils other do


lib_LIBRARIES 		= libeo.a

libeo_a_SOURCES 	= eoFunctorStore.cpp \
			eoPersistent.cpp \
			eoPrintable.cpp \
			eoCtrlCContinue.cpp \
			eoParetoFitness.cpp \
			eoScalarFitnessAssembled.cpp  

pkginclude_HEADERS 	= $(srcdir)/eo $(srcdir)/*.h $(srcdir)/do/*.h 


AM_CXXFLAGS		= -I$(top_srcdir)/src

EXTRA_DIST		= eo *.h do/*.h
