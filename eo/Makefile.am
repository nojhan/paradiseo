###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS 	= src test doc app tutorial contrib win


#Directory for documents
DOCDIR 		= ~/public_html/eodocs

#Directory for indices -- not useful for the user
IDXDIR 		= ~/index


dist-hook:
		$(MAKE) doc

doc:    	$(srcdir)/doc/eo.cfg
		cd doc && $(MAKE) doc && cd ..
