
noinst_PROGRAMS		= BitEA RealEA ESEA


BitEA_SOURCES		= BitEA.cpp

RealEA_SOURCES		= RealEA.cpp

ESEA_SOURCES		= ESEA.cpp


noinst_HEADERS		= binary_value.h \
			real_value.h

extra_DIST		= Makefile.simple


LDADD			= -L$(top_builddir)/src -L$(top_builddir)/src/es \
			-L$(top_builddir)/src/ga -L$(top_builddir)/src/utils

LIBS			=  -lga -les -leoutils -leo

INCLUDES		= -I$(top_srcdir)/src
