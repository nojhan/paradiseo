# Makefile.am for app/gpsymreg

noinst_PROGRAMS 	= gpsymreg

gpsymreg_SOURCES 	= main.cpp

noinst_HEADERS		= fitness.h node.h parameters.h


LIBEO			= $(top_builddir)/src/libeo.a
LIBEOUTILS		= $(top_builddir)/src/utils/libeoutils.a

AM_CXXFLAGS		= -I$(top_srcdir)/src
DEPS 			= $(LIBEOUTILS) $(LIBEO)
LIBS 			= $(LIBEOUTILS) $(LIBEO)
