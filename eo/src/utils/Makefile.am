###############################################################################
##
## Makefile.am for eo/src/utils
##
###############################################################################

lib_LIBRARIES = libeoutils.a

libeoutils_a_SOURCES 	= eoParser.cpp \
			eoRNG.cpp \
			eoState.cpp \
			eoUpdater.cpp \
			eoFileMonitor.cpp \
			eoStdoutMonitor.cpp \
			eoRealBounds.cpp \
			make_help.cpp

pkginclude_HEADERS 	= checkpointing $(srcdir)/*.h

AM_CXXFLAGS 		= -I$(top_srcdir)/src

EXTRA_DIST		= *.h
