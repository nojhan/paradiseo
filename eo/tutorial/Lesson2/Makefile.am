
noinst_PROGRAMS		= FirstBitEA FirstRealEA exercise2.3


FirstBitEA_SOURCES	= FirstBitEA.cpp

FirstRealEA_SOURCES	= FirstRealEA.cpp

exercise2_3_SOURCES	= exercise2.3.cpp


noinst_HEADERS		= binary_value.h \
			real_value.h

extra_DIST		= Makefile.simple


LDADD			= -L$(top_builddir)/src -L$(top_builddir)/src/ga -L$(top_builddir)/src/utils

LIBS			=  -lga -leoutils -leo

INCLUDES		= -I$(top_srcdir)/src

