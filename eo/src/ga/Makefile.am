###############################################################################
##
## Makefile.am for eo/src/ga
##
###############################################################################

INCLUDES = -I$(top_builddir)/src
lib_LIBRARIES = libga.a
libga_a_SOURCES = make_algo_scalar_ga.cpp \
                  make_checkpoint_ga.cpp \
                  make_continue_ga.cpp \
                  make_genotype_ga.cpp \
                  make_help.cpp \
                  make_op_ga.cpp \
                  make_pop_ga.cpp \
                  make_run_ga.cpp
CPPFLAGS = -Wall
CXXFLAGS = -g
libeoincdir = $(includedir)/eo/ga
libeoinc_HEADERS = eoBit.h eoBitOp.h eoBitOpFactory.h make_ga.h
