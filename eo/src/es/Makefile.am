###############################################################################
##
## Makefile.am for eo/src/es
##
###############################################################################

INCLUDES = -I$(top_builddir)/src
lib_LIBRARIES = libes.a
libes_a_SOURCES = make_algo_scalar_real.cpp make_checkpoint_real.cpp \
  make_continue_real.cpp make_genotype_real.cpp \
  make_op_real.cpp make_pop_real.cpp make_run_real.cpp \
  make_algo_scalar_es.cpp make_checkpoint_es.cpp \
  make_continue_es.cpp make_genotype_es.cpp \
  make_op_es.cpp make_pop_es.cpp make_run_es.cpp

CPPFLAGS = -Wall
CXXFLAGS = 
libeoincdir = $(includedir)/eo/es
libeoinc_HEADERS = eoEsChromInit.h eoEsFull.h eoEsMutate.h eoEsMutationInit.h eoEsSimple.h eoEsStdev.h  eoNormalMutation.h eoReal.h eoRealOp.h eoEsGlobalXover.h eoEsStandardXover.h eoRealAtomXover.h
