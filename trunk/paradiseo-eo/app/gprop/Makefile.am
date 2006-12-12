# Makefile.am for app/gprop

noinst_PROGRAMS 	= gprop


gprop_SOURCES 		= gprop.cpp

noinst_HEADERS		= gprop.h \
			l2.h \
			mlp.h \
			mse.h \
			qp.h \
			vecop.h


AM_CXXFLAGS		= -I$(top_srcdir)/src

LIBEO			= $(top_builddir)/src/libeo.a
LIBEOUTILS		= $(top_builddir)/src/utils/libeoutils.a

DEPS 			= $(LIBEO) $(LIBEOUTILS)
LIBS 			= $(LIBEO) $(LIBEOUTILS)
