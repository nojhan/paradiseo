###############################################################################
#
# Makefile.am for app/gpsymreg
#
###############################################################################

DEPS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

bin_PROGRAMS = gpsymreg

###############################################################################

gpsymreg_SOURCES = main.cpp
gpsymreg_DEPENDENCIES = $(DEPS)
gpsymreg_LDFLAGS = -lm
gpsymreg_LDADD = $(LDADDS)

###############################################################################
