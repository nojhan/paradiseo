
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

LIBEO			= $(top_builddir)/src/libeo.a
LIBES			= $(top_builddir)/src/es/libes.a
LIBGA			= $(top_builddir)/src/ga/libga.a
LIBUTILS		= $(top_builddir)/src/utils/libeoutils.a

DEPS			= $(LIBEO) $(LIBUTILS) $(LIBES) $(LIBGA)

LIBS			= $(LIBES) $(LIBGA) $(LIBEO) $(LIBUTILS)
