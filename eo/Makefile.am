###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test win app tutorial 
#Directory for documents
DOCDIR = ~/public_html/eodocs
#Directory for indices -- not useful for the user
IDXDIR = ~/index

EXTRA_DIST=LICENSE

###############################################################################

lib:	
	cd src; $(MAKE) all; cd ..

# so that make doc always compiles the doc ...
doc:    doc/eo.cfg
	cd doc; $(MAKE) doc; touch eo.cfg; cd ..

tutorial:
	cd tutorial; $(MAKE) all; cd ..

app:
	cd app; $(MAKE) all; cd ..

all:
	for i in $(SUBDIRS); do cd $$i && $(MAKE) all; cd ..; done

clean:
	for i in $(SUBDIRS); do cd $$i && $(MAKE) clean; cd ..; done

###############################################################################
