noinst_PROGRAMS			= IslandBitEA IslandBitEA1 IslandBitEA2

LIBEO				= $(top_builddir)/src/libeo.a
LIBEOUTILS			= $(top_builddir)/src/utils/libeoutils.a

CXX				= $(MPICXX)
LD				= $(MPICXX)
AM_CXXFLAGS			= -I$(top_srcdir)/src
DEPS				= $(LIBEOUTILS) $(LIBEO)
LIBS				= $(LIBEOUTILS) $(LIBEO)


noinst_HEADERS			= binary_value.h

IslandBitEA_SOURCES		= IslandBitEA.cpp

IslandBitEA1_SOURCES		= IslandBitEA1.cpp

IslandBitEA2_SOURCES            = IslandBitEA2.cpp
