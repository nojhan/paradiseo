# Makefile.am for app/mastermind

noinst_PROGRAMS 	= mastermind

mastermind_SOURCES 	= mastermind.cpp

noinst_HEADERS		= mastermind.h

LIBEO			= $(top_builddir)/src/libeo.a
LIBEOUTILS		= $(top_builddir)/src/utils/libeoutils.a

AM_CXXFLAGS		= -I$(top_srcdir)/src
DEPS 			= $(LIBEOUTILS) $(LIBEO)
LIBS			= $(LIBEOUTILS) $(LIBEO)
