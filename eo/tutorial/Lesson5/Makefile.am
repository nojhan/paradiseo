
noinst_PROGRAMS		= OneMaxEA OneMaxLibEA

OneMaxEA_SOURCES	= OneMaxEA.cpp

OneMaxLibEA_SOURCES	= OneMaxLibEA.cpp make_OneMax.cpp


noinst_HEADERS		= eoOneMax.h \
			eoOneMaxEvalFunc.h \
			eoOneMaxInit.h \
			eoOneMaxMutation.h \
			eoOneMaxQuadCrossover.h \
			make_genotype_OneMax.h \
			make_op_OneMax.h

extra_DIST		= Makefile.simple

AM_CXXFLAGS		= -I$(top_srcdir)/src
LDADD			= -L$(top_builddir)/src -L$(top_builddir)/src/ga -L$(top_builddir)/src/utils
LIBS			=  -lga -leoutils -leo

