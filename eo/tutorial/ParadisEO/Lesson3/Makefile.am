noinst_PROGRAMS			= CellularBitEA

LIBEO				= $(top_builddir)/src/libeo.a
LIBEOUTILS			= $(top_builddir)/src/utils/libeoutils.a

CXX				= $(MPICXX)
LD				= $(MPICXX)
AM_CXXFLAGS			= -I$(top_srcdir)/src
DEPS				= $(LIBEOUTILS) $(LIBEO)
LIBS				= $(LIBEOUTILS) $(LIBEO)


noinst_HEADERS			= binary_value.h

CellularBitEA_SOURCES		= CellularBitEA.cpp
