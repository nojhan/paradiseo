###############################################################################
##
## Makefile.am for eo/test
##
###############################################################################

DEPS = $(top_builddir)/src/utils/libeoutils.a $(top_builddir)/src/libeo.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/utils/libeoutils.a $(top_builddir)/src/libeo.a
CXXFLAGS = -g -Wall 
###############################################################################

check_PROGRAMS = t-eoParetoFitness t-eoPareto t-eofitness t-eoRandom t-eobin t-eoVirus t-MGE t-MGE-control t-eoStateAndParser t-eoCheckpointing t-eoSSGA  \
	t-eoExternalEO t-eoSymreg t-eo t-eoReplacement t-eoSelect t-eoGenOp t-eoGA t-eoReal t-eoVector t-eoESAll
TESTS=run_tests t-eoVector t-eoRandom t-eoSSGA t-eoPareto t-eoParetoFitness
# removing temporarily t-eoESFull
# noinst_PROGRAMS = t-eofitness t-eobin t-eoStateAndParser t-eoCheckpointing t-eoExternalEO t-eoESFull t-eoSymreg t-eo t-eoReplacement t-eoSelect t-eoGenOp t-eoGA t-eoReal

###############################################################################

t_eoRandom_SOURCES = t-eoRandom.cpp
t_eoRandom_DEPENDENCIES = $(DEPS)
t_eoRandom_LDADD = $(LDADDS)

###############################################################################

t_eofitness_SOURCES = t-eofitness.cpp
t_eofitness_DEPENDENCIES = $(DEPS)
t_eofitness_LDADD = $(LDADDS)

###############################################################################

t_eobin_SOURCES = t-eobin.cpp binary_value.h
t_eobin_DEPENDENCIES = $(DEPS)
t_eobin_LDADD = $(LDADDS)

###############################################################################

t_eoVirus_SOURCES = t-eoVirus.cpp binary_value.h
t_eoVirus_DEPENDENCIES = $(DEPS)
t_eoVirus_LDADD = $(LDADDS)

###############################################################################

t_MGE_SOURCES = t-MGE.cpp binary_value.h
t_MGE_DEPENDENCIES = $(DEPS)
t_MGE_LDADD = $(LDADDS)

###############################################################################

t_MGE_control_SOURCES = t-MGE-control.cpp binary_value.h
t_MGE_control_DEPENDENCIES = $(DEPS)
t_MGE_control_LDADD = $(LDADDS)

###############################################################################

t_eoStateAndParser_SOURCES = t-eoStateAndParser.cpp
t_eoStateAndParser_DEPENDENCIES = $(DEPS)
t_eoStateAndParser_LDFLAGS = -lm
t_eoStateAndParser_LDADD = $(LDADDS)

###############################################################################

t_eoCheckpointing_SOURCES = t-eoCheckpointing.cpp
t_eoCheckpointing_DEPENDENCIES = $(DEPS)
t_eoCheckpointing_LDFLAGS = -lm
t_eoCheckpointing_LDADD = $(LDADDS)

###############################################################################

t_eoReplacement_SOURCES = t-eoReplacement.cpp
t_eoReplacement_DEPENDENCIES = $(DEPS)
t_eoReplacement_LDFLAGS = -lm
t_eoReplacement_LDADD = $(LDADDS)

###############################################################################

t_eoSelect_SOURCES = t-eoSelect.cpp
t_eoSelect_DEPENDENCIES = $(DEPS)
t_eoSelect_LDFLAGS = -lm
t_eoSelect_LDADD = $(LDADDS)

###############################################################################

t_eoExternalEO_SOURCES = t-eoExternalEO.cpp
t_eoExternalEO_DEPENDENCIES = $(DEPS)
t_eoExternalEO_LDFLAGS = -lm
t_eoExternalEO_LDADD = $(LDADDS)

###############################################################################

t_eoSymreg_SOURCES = t-eoSymreg.cpp
t_eoSymreg_DEPENDENCIES = $(DEPS)
t_eoSymreg_LDFLAGS = -lm
t_eoSymreg_LDADD = $(LDADDS)

###############################################################################

t_eo_SOURCES = t-eo.cpp
t_eo_DEPENDENCIES = $(DEPS)
t_eo_LDFLAGS = -lm
t_eo_LDADD = $(LDADDS)

###############################################################################

t_eoGenOp_SOURCES = t-eoGenOp.cpp
t_eoGenOp_DEPENDENCIES = $(DEPS)
t_eoGenOp_LDFLAGS = -lm
t_eoGenOp_LDADD = $(LDADDS)

###############################################################################

t_eoVector_SOURCES = t-eoVector.cpp
t_eoVector_DEPENDENCIES = $(DEPS)
t_eoVector_LDFLAGS = -lm
t_eoVector_LDADD = $(LDADDS)

###############################################################################

t_eoGA_SOURCES = t-eoGA.cpp binary_value.h
t_eoGA_DEPENDENCIES = $(DEPS) $(top_builddir)/src/ga/libga.a
t_eoGA_LDFLAGS = -lm
t_eoGA_LDADD = $(top_builddir)/src/ga/libga.a $(LDADDS)

###############################################################################

t_eoReal_SOURCES = t-eoReal.cpp real_value.h
t_eoReal_DEPENDENCIES = $(DEPS) $(top_builddir)/src/es/libes.a
t_eoReal_LDFLAGS = -lm
t_eoReal_LDADD = $(top_builddir)/src/es/libes.a $(LDADDS)

###############################################################################

t_eoESAll_SOURCES = t-eoESAll.cpp real_value.h
t_eoESAll_DEPENDENCIES = $(DEPS) $(top_builddir)/src/es/libes.a
t_eoESAll_LDFLAGS = -lm
t_eoESAll_LDADD = $(top_builddir)/src/es/libes.a $(LDADDS)
###############################################################################

t_eoSSGA_SOURCES = t-eoSSGA.cpp binary_value.h
t_eoSSGA_DEPENDENCIES = $(DEPS) $(top_builddir)/src/ga/libga.a
t_eoSSGA_LDFLAGS = -lm
t_eoSSGA_LDADD = $(LDADDS)

###############################################################################

t_eoPareto_SOURCES = t-eoPareto.cpp 
t_eoPareto_DEPENDENCIES = $(DEPS) $(top_builddir)/src/ga/libga.a
t_eoPareto_LDFLAGS = -lm
t_eoPareto_LDADD = $(LDADDS)
###############################################################################

t_eoParetoFitness_SOURCES = t-eoParetoFitness.cpp 
t_eoParetoFitness_DEPENDENCIES = $(DEPS) $(top_builddir)/src/ga/libga.a
t_eoParetoFitness_LDFLAGS = -lm
t_eoParetoFitness_LDADD = $(LDADDS)
###############################################################################
