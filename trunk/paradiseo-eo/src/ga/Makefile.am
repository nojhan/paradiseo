###############################################################################
##
## Makefile.am for eo/src/ga
##
###############################################################################

AM_CXXFLAGS		= -I$(top_builddir)/src  -I$(top_srcdir)/src

lib_LIBRARIES 		= libga.a

libga_a_SOURCES 	= make_algo_scalar_ga.cpp \
			make_checkpoint_ga.cpp \
			make_continue_ga.cpp \
			make_genotype_ga.cpp \
			make_op_ga.cpp \
			make_pop_ga.cpp \
			make_run_ga.cpp

gaincludedir 		= $(pkgincludedir)/ga

gainclude_HEADERS 	= eoBit.h \
			eoBitOpFactory.h \
			eoBitOp.h \
			eoBoolFlip.h \
			eoPBILAdditive.h \
			eoPBILDistrib.h \
			eoPBILOrg.h \
			make_ga.h \
			make_genotype_ga.h \
			make_op.h \
			make_PBILdistrib.h \
			make_PBILupdate.h
