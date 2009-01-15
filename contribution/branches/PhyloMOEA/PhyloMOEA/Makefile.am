INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/eo -I$(top_srcdir)/mo \
	-I$(top_srcdir)/moeo -I$(top_srcdir)/moeo/core
METASOURCES = AUTO
bin_PROGRAMS = PhyloMOEA
PhyloMOEA_SOURCES = ExceptionManager.h Patterns.h PhyloMOEA.cpp ProbMatrix.cpp \
	ProbMatrix.h RandomNr.cpp RandomNr.h Sequences.cpp Sequences.h SubsModel.cpp \
	SubsModel.h eigensolver.cpp eigensolver.h likelihoodcalculator.cpp \
	likelihoodcalculator.h likoptimizer.cpp likoptimizer.h matrixutils.cpp matrixutils.h \
	parsimonycalculator.cpp parsimonycalculator.h phylotreeIND.cpp phylotreeIND.h \
	probmatrixcontainer.cpp probmatrixcontainer.h treeIterator.cpp treeIterator.h tree_limits.h \
	utils.cpp
PhyloMOEA_LDADD = $(top_builddir)/moeo/core/libmoeo.la \
	$(top_builddir)/eo/utils/libutils.la $(top_builddir)/eo/ga/libga.la $(top_builddir)/eo/libeo.la -lGTL -lgsl \
	-lgslcblas
noinst_HEADERS = PhyloMOEOProbMatrixContainerUpdater.h moeoObjVecStat.h \
	vectorSortIndex.h
_SOURCES = PhyloMOEO.h
_SOURCES = PhyloMOEO.h PhyloMOEO_operators.h
_SOURCES = PhyloMOEO.h PhyloMOEO_init.h PhyloMOEO_operators.h
_SOURCES = PhyloMOEO.h PhyloMOEO_eval.h PhyloMOEO_init.h PhyloMOEO_operators.h
_SOURCES = PhyloMOEO.h PhyloMOEO_eval.h PhyloMOEO_init.h PhyloMOEO_operators.h \
	utils.h
_SOURCES = eoCountedFileMonitor.h
_SOURCES = eoCountedFileMonitor.h eoSingleFileCountedStateSaver.h
_SOURCES = moeoPtrComparator.h
_SOURCES = PhyloMOEO_archive.h
_SOURCES = PhyloMOEOPartitionStat.h PhyloMOEO_archive.h
