###############################################################################
#
# Makefile.am for app/gprop
#
###############################################################################

DEPS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

bin_PROGRAMS = gprop

###############################################################################

gprop_SOURCES = gprop.cc
gprop_DEPENDENCIES = $(DEPS)
gprop_LDFLAGS = -lm
gprop_LDADD = $(LDADDS)

###############################################################################
