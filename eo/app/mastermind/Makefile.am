###############################################################################
#
# Makefile.am for app/mastermind
#
###############################################################################

DEPS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/libeo.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

bin_PROGRAMS = mastermind

###############################################################################

mastermind_SOURCES = mastermind.cpp
mastermind_DEPENDENCIES = $(DEPS)
mastermind_LDFLAGS = -lm
mastermind_LDADD = $(LDADDS)

###############################################################################
