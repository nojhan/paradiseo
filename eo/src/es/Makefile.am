## Makefile.am for eo/src/es

lib_LIBRARIES 		= libes.a

libes_a_SOURCES 	= make_algo_scalar_es.cpp \
			make_algo_scalar_real.cpp \
			make_checkpoint_es.cpp \
			make_checkpoint_real.cpp \
			make_continue_es.cpp \
			make_continue_real.cpp \
			make_genotype_es.cpp \
			make_genotype_real.cpp \
			make_op_es.cpp \
			make_op_real.cpp \
			make_pop_es.cpp \
			make_pop_real.cpp \
			make_run_es.cpp \
			make_run_real.cpp

esincludedir		= $(pkgincludedir)/es

esinclude_HEADERS 	= eoEsChromInit.h \
			eoEsFull.h \
			eoEsGlobalXover.h \
			eoEsMutate.h \
			eoEsMutationInit.h \
			eoEsSimple.h \
			eoEsStandardXover.h \
			eoEsStdev.h \
			eoNormalMutation.h \
			eoRealAtomXover.h \
			eoReal.h \
			eoRealInitBounded.h \
			eoRealOp.h \
			eoSBXcross.h \
			make_es.h \
			make_genotype_real.h \
			make_op_es.h \
			make_op.h \
			make_op_real.h \
			make_real.h


AM_CXXFLAGS		= -I$(top_srcdir)/src
