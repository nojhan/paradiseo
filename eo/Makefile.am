###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test win doc tutorial app
#Directory for documents
DOCDIR = ~/public_html/eodocs
#Directory for indices -- not useful for the user
IDXDIR = ~/index

EXTRA_DIST=LICENSE

###############################################################################

lib:
	pushd src; $(MAKE) all; popd

test:
	pushd test; $(MAKE) all; popd

doc:
	pushd doc; $(MAKE) doc; popd

tutorial:
	pushd tutorial; $(MAKE) all; popd

app:
	pushd app; $(MAKE) all; popd

all:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) all; popd; done

clean:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) clean; popd; done

###############################################################################