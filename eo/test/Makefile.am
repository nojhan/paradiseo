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

noinst_PROGRAMS = t-eobreeder t-eoinclusion t-eoinsertion t-eo t-eofitness t-eoproblem t-eobin t-eolottery t_eornd t_eovector t_eoaged t_eoid t_eostring t_ops t_popops t_pop t_es t_opsel t_opfactory t_opMut t_eobitfact

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
t_eovector_LDFLAGS = -lm
t_eolottery_LDADD = $(LDADDS) 

###############################################################################

t_eornd_SOURCES = t_eornd.cpp
t_eornd_LDFLAGS = -lm

###############################################################################

t_eovector_SOURCES = t_eovector.cpp
t_eovector_LDFLAGS = -lm
t_eovector_LDADD = $(LDADDS)

###############################################################################

t_eoaged_SOURCES = t_eoaged.cpp
t_eoaged_DEPENDENCIES = $(DEPS)
t_eoaged_LDADD = $(LDADDS)

###############################################################################

t_eoid_SOURCES = t_eoid.cpp
t_eoid_DEPENDENCIES = $(DEPS)
t_eoid_LDADD = $(LDADDS)

###############################################################################

t_eostring_SOURCES = t_eostring.cpp
t_eostring_DEPENDENCIES = $(DEPS)
t_eostring_LDADD = $(LDADDS)

###############################################################################

t_eostring_SOURCES = t_eostring.cpp
t_eostring_DEPENDENCIES = $(DEPS)
t_eostring_LDADD = $(LDADDS)

###############################################################################

t_es_SOURCES = t_es.cpp
t_es_DEPENDENCIES = $(DEPS)
t_es_LDADD = $(LDADDS)

###############################################################################

t_ops_SOURCES = t_ops.cpp
t_ops_DEPENDENCIES = $(DEPS)
t_ops_LDADD = $(LDADDS)

###############################################################################

t_pop_SOURCES = t_pop.cpp
t_pop_DEPENDENCIES = $(DEPS)
t_pop_LDADD = $(LDADDS)

###############################################################################

t_popops_SOURCES = t_popops.cpp
t_popops_DEPENDENCIES = $(DEPS)
t_popops_LDADD = $(LDADDS)

###############################################################################

t_opsel_SOURCES = t_opsel.cpp
t_opsel_DEPENDENCIES = $(DEPS)
t_opsel_LDADD = $(LDADDS)

###############################################################################

t_opfactory_SOURCES = t_opfactory.cpp
t_opfactory_DEPENDENCIES = $(DEPS)
t_opfactory_LDADD = $(LDADDS)

###############################################################################

t_opMut_SOURCES = t_opMut.cpp
t_opMut_DEPENDENCIES = $(DEPS)
t_opMut_LDADD = $(LDADDS)

###############################################################################

t_eobitfact_SOURCES = t_eobitfact.cpp
t_eobitfact_DEPENDENCIES = $(DEPS)
t_eobitfact_LDADD = $(LDADDS)

###############################################################################
