
noinst_PROGRAMS		= SecondBitEA SecondRealEA exercise3.1


SecondBitEA_SOURCES	= SecondBitEA.cpp

SecondRealEA_SOURCES	= SecondRealEA.cpp

exercise3_1_SOURCES	= exercise3.1.cpp


noinst_HEADERS		= binary_value.h \
			real_value.h

extra_DIST		= Makefile.simple


LDADD			= -L$(top_builddir)/src -L$(top_builddir)/src/ga -L$(top_builddir)/src/utils

LIBS			=  -lga -leoutils -leo

INCLUDES		= -I$(top_srcdir)/src
