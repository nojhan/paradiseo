###############################################################################
##
## Makefile.am for eo/src/es
##
###############################################################################

INCLUDES = -I$(top_builddir)/src
lib_LIBRARIES = libes.a
libes_a_SOURCES = make_algo_scalar_real.cpp \
                  make_checkpoint_real.cpp \
                  make_continue_real.cpp \
                  make_genotype_real.cpp \
                  make_help.cpp \
                  make_op_real.cpp \
                  make_pop_real.cpp \
                  make_run_real.cpp
CPPFLAGS = -Wall
CXXFLAGS = -g
libeoincdir = $(includedir)/eo/es
libeoinc_HEADERS = eoEsChromInit.h eoEsFull.h eoEsMutate.h eoEsMutationInit.h eoEsSimple.h eoEsStdev.h  eoEsStdevXOver.h eoGenericRealOp.h eoNormalMutation.h eoReal.h eoRealOp.h
