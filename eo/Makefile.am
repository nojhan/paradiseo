###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test win doc app
#Directory for documents
DOCDIR = ~/public_html/eodocs
#Directory for indices -- not useful for the user
IDXDIR = ~/index

EXTRA_DIST=LICENSE

###############################################################################

all:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) all; popd; done

clean:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) clean; popd; done

###############################################################################