###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test doc app tutorial contrib win


#Directory for documents
DOCDIR = ~/public_html/eodocs

#Directory for indices -- not useful for the user
IDXDIR = ~/index


# special targets
app:
	cd app; $(MAKE) all; cd ..

# so that make doc always compiles the doc ...
doc:    doc/eo.cfg
	cd doc; $(MAKE) doc; touch eo.cfg; cd ..




###############################################################################

#lib:	
#	cd src; $(MAKE) all; cd ..
