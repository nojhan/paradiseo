lib_LIBRARIES			= libflowshop.a

libflowshop_a_SOURCES	= \
						FlowShop.cpp \
						FlowShopBenchmarkParser.cpp \
						FlowShopEval.cpp \
						FlowShopInit.cpp \
						FlowShopObjectiveVectorTraits.cpp \
						FlowShopOpCrossoverQuad.cpp \
						FlowShopOpMutationExchange.cpp \
						FlowShopOpMutationShift.cpp

pkginclude_HEADERS		= \
						FlowShop.h \
						FlowShopBenchmarkParser.h \
						FlowShopEval.h \
						FlowShopInit.h \
						FlowShopObjectiveVector.h \
						FlowShopObjectiveVectorTraits.h \
						FlowShopOpCrossoverQuad.h \
						FlowShopOpMutationExchange.h \
						FlowShopOpMutationShift.h

INCLUDES				= -I$(EO_DIR)/src/ -I$(top_srcdir)/src/

AM_CXXFLAGS				= -Wall -ansi -pedantic
