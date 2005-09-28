noinst_PROGRAMS		= BitEA RealEA ESEA


BitEA_SOURCES		= BitEA.cpp

RealEA_SOURCES		= RealEA.cpp

ESEA_SOURCES		= ESEA.cpp


noinst_HEADERS		= binary_value.h \
			real_value.h

extra_DIST		= Makefile.simple


AM_CXXFLAGS		= -I$(top_srcdir)/src

LIBEO			= $(top_builddir)/src/libeo.a
LIBES			= $(top_builddir)/src/es/libes.a
LIBGA			= $(top_builddir)/src/ga/libga.a
LIBUTILS		= $(top_builddir)/src/utils/libeoutils.a

DEPS			= $(LIBEO) $(LIBUTILS) $(LIBES) $(LIBGA)

LIBS			= $(LIBES) $(LIBGA) $(LIBEO) $(LIBUTILS)
