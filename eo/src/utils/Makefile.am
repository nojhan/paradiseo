###############################################################################
##
## Makefile.am for eo/src/utils
##
###############################################################################

lib_LIBRARIES = libeoutils.a


AM_CXXFLAGS 		= -I$(top_srcdir)/src


libeoutils_a_SOURCES 	= eoParser.cpp \
			eoRNG.cpp \
			eoState.cpp \
			eoUpdater.cpp \
			eoFileMonitor.cpp \
			eoStdoutMonitor.cpp \
			eoRealBounds.cpp \
			make_help.cpp

libeoincdir 		= $(includedir)/eo/utils

libeoinc_HEADERS 	= checkpointing $(srcdir)/*.h
