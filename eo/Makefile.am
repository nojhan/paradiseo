###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS 	= src test doc contrib win
if USE_APPLICATIONS
SUBDIRS		+= app
endif
if USE_TUTORIAL
SUBDIRS		+= tutorial
endif

# Directory for documents
DOCDIR 		= ~/public_html/eodocs

#Directory for indices -- not useful for the user
IDXDIR 		= ~/index


dist-hook:
		$(MAKE) doc

doc:    	$(srcdir)/doc/eo.cfg
		cd doc && $(MAKE) doc && cd ..
