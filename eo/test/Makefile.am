###############################################################################
##
## Makefile.am for eo/test
##
###############################################################################

DEPS = $(top_builddir)/src/libeo.a $(top_builddir)/src/obsolete/libeoobsolete.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

INCLUDES = -I$(top_builddir)/src
LDADDS = $(top_builddir)/src/libeo.a $(top_builddir)/src/obsolete/libeoobsolete.a $(top_builddir)/src/utils/libeoutils.a

###############################################################################

noinst_PROGRAMS = t-eobreeder t-eoinclusion t-eoinsertion t-eo t-eofitness \
	t-eoproblem t-eobin t-eolottery t-eo2dVector t-eogeneration t-eoEasyEA\
	t-eoNonUniform t-eoUniform t-eoRandom t-eoStateAndParser \
	t-eoAtomOps t-selectOne t-eoGOpSel \
	t-eoVector t-eoCheckpointing t-eoExternalEO t-eoESFull

###############################################################################

t_eoVector_SOURCES = t-eoVector.cpp
t_eoVector_DEPENDENCIES = $(DEPS)
t_eoVector_LDFLAGS = -lm
t_eoVector_LDADD = $(LDADDS)

###############################################################################

t_eoGOpSel_SOURCES = t-eoGOpSel.cpp
t_eoGOpSel_DEPENDENCIES = $(DEPS)
t_eoGOpSel_LDFLAGS = -lm
t_eoGOpSel_LDADD = $(LDADDS)

###############################################################################

t_selectOne_SOURCES = t-selectOne.cpp
t_selectOne_DEPENDENCIES = $(DEPS)
t_selectOne_LDFLAGS = -lm
t_selectOne_LDADD = $(LDADDS)

###############################################################################

t_eoAtomOps_SOURCES = t-eoAtomOps.cpp
t_eoAtomOps_DEPENDENCIES = $(DEPS)
t_eoAtomOps_LDFLAGS = -lm
t_eoAtomOps_LDADD = $(LDADDS)

###############################################################################
#disabled
#t_eoESOps_SOURCES = t-eoESOps.cpp
#t_eoESOps_DEPENDENCIES = $(DEPS)
#t_eoESOps_LDFLAGS = -lm
#t_eoESOps_LDADD = $(LDADDS)

###############################################################################

t_eoESFull_SOURCES = t-eoESFull.cpp real_value.h
t_eoESFull_DEPENDENCIES = $(DEPS)
t_eoESFull_LDFLAGS = -lm
t_eoESFull_LDADD = $(LDADDS)

###############################################################################

t_eoNonUniform_SOURCES = t-eoNonUniform.cpp
t_eoNonUniform_DEPENDENCIES = $(DEPS)
t_eoNonUniform_LDFLAGS = -lm
t_eoNonUniform_LDADD = $(LDADDS)

###############################################################################

t_eoUniform_SOURCES = t-eoUniform.cpp
t_eoUniform_DEPENDENCIES = $(DEPS)
t_eoUniform_LDFLAGS = -lm
t_eoUniform_LDADD = $(LDADDS)

###############################################################################

t_eoRandom_SOURCES = t-eoRandom.cpp
t_eoRandom_DEPENDENCIES = $(DEPS)
t_eoRandom_LDFLAGS = -lm
t_eoRandom_LDADD = $(LDADDS)

###############################################################################

t_eogeneration_SOURCES = t-eogeneration.cpp
t_eogeneration_DEPENDENCIES = $(DEPS)
t_eogeneration_LDFLAGS = -lm
t_eogeneration_LDADD = $(LDADDS)

###############################################################################

t_eoEasyEA_SOURCES = t-eoEasyEA.cpp binary_value.h
t_eoEasyEA_DEPENDENCIES = $(DEPS)
t_eoEasyEA_LDFLAGS = -lm
t_eoEasyEA_LDADD = $(LDADDS)

###############################################################################

t_eobreeder_SOURCES = t-eobreeder.cpp binary_value.h
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

t_eoExternalEO_SOURCES = t-eoExternalEO.cpp
t_eoExternalEO_DEPENDENCIES = $(DEPS) 
t_eoExternalEO_LDFLAGS = -lm 
t_eoExternalEO_LDADD = $(LDADDS) 

###############################################################################


