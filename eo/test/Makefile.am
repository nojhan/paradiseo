###############################################################################
##
## Makefile.am for eo/test
##
###############################################################################

DEPS = $(top_builddir)/src/libeo.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/libeo.a

###############################################################################

noinst_PROGRAMS = t-eobreeder t-eoinclusion t-eoinsertion t-eo t-eofitness t-eoproblem t-eobin t-eolottery t-eo2dVector t-eogeneration t-eoEasyEA t-eoNonUniform
###############################################################################

t_eoNonUniform_SOURCES = t-eoNonUniform.cpp
t_eoNonUniform_DEPENDENCIES = $(DEPS)
t_eoNonUniform_LDFLAGS = -lm
t_eoNonUniform_LDADD = $(LDADDS)

###############################################################################

t_eogeneration_SOURCES = t-eogeneration.cpp
t_eogeneration_DEPENDENCIES = $(DEPS)
t_eogeneration_LDFLAGS = -lm
t_eogeneration_LDADD = $(LDADDS)

###############################################################################

t_eoEasyEA_SOURCES = t-eoEasyEA.cpp
t_eoEasyEA_DEPENDENCIES = $(DEPS)
t_eoEasyEA_LDFLAGS = -lm
t_eoEasyEA_LDADD = $(LDADDS)

###############################################################################

t_eobreeder_SOURCES = t-eobreeder.cpp
t_eobreeder_DEPENDENCIES = $(DEPS)
t_eobreeder_LDFLAGS = -lm
t_eobreeder_LDADD = $(LDADDS)

###############################################################################

t_eoinclusion_SOURCES = t-eoinclusion.cpp
t_eoinclusion_DEPENDENCIES = $(DEPS)
t_eoinclusion_LDFLAGS = -lm
t_eoinclusion_LDADD = $(LDADDS)

###############################################################################

t_eoinsertion_SOURCES = t-eoinsertion.cpp
t_eoinsertion_DEPENDENCIES = $(DEPS)
t_eoinsertion_LDFLAGS = -lm
t_eoinsertion_LDADD = $(LDADDS)

###############################################################################

t_eo_SOURCES = t-eo.cpp
t_eo_DEPENDENCIES = $(DEPS)
t_eo_LDADD = $(LDADDS)

###############################################################################

t_eofitness_SOURCES = t-eofitness.cpp
t_eofitness_DEPENDENCIES = $(DEPS)
t_eofitness_LDADD = $(LDADDS)

###############################################################################

t_eoproblem_SOURCES = t-eoproblem.cpp
t_eoproblem_LDFLAGS = -lm

###############################################################################

t_eobin_SOURCES = t-eobin.cpp
t_eobin_DEPENDENCIES = $(DEPS)
t_eobin_LDADD = $(LDADDS)

###############################################################################

t_eolottery_SOURCES = t-eolottery.cpp
t_eolottery_DEPENDENCIES = $(DEPS)
t_eolottery_LDFLAGS = -lm
t_eolottery_LDADD = $(LDADDS) 

###############################################################################

t_eo2dVector_SOURCES = t-eo2dVector.cc
t_eo2dVector_DEPENDENCIES = $(DEPS) 
t_eo2dVector_LDFLAGS = -lm 
t_eo2dVector_LDADD = $(LDADDS) 

###############################################################################
