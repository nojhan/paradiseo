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

noinst_PROGRAMS = t-eofitness t-eobin t-eoStateAndParser t-eoCheckpointing t-eoExternalEO t-eoESFull

###############################################################################

t_eoESFull_SOURCES = t-eoESFull.cpp real_value.h
t_eoESFull_DEPENDENCIES = $(DEPS)
t_eoESFull_LDFLAGS = -lm
t_eoESFull_LDADD = $(LDADDS)

###############################################################################

t_eofitness_SOURCES = t-eofitness.cpp
t_eofitness_DEPENDENCIES = $(DEPS)
t_eofitness_LDADD = $(LDADDS)

###############################################################################

t_eobin_SOURCES = t-eobin.cpp
t_eobin_DEPENDENCIES = $(DEPS)
t_eobin_LDADD = $(LDADDS)

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


