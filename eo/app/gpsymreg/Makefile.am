###############################################################################
#
# Makefile.am for app/gpsymreg
#
###############################################################################

LIBEO		= $(top_builddir)/src/libeo.a
LIBEOUTILS	= $(top_builddir)/src/utils/libeoutils.a


bin_PROGRAMS = gpsymreg

AM_CXXFLAGS	= -I$(top_builddir)/src
DEPS 		= $(LIBEOUTILS) $(LIBEO)
LIBS 		=  $(top_builddir)/src/utils/libeoutils.a

gpsymreg_SOURCES = main.cpp
