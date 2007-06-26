SUBDIRS				= flowshop

noinst_PROGRAMS		= FlowShopEA

FlowShopEA_SOURCES	= FlowShopEA.cpp

LDADD				= ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a $(top_srcdir)/src/core/libmoeo.a flowshop/libflowshop.a

INCLUDES			= -Iflowshop -I${EO_DIR}/src/ -I$(top_srcdir)/src/

AM_CXXFLAGS			= -Wall -ansi -pedantic
