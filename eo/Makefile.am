###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src win tutorial app
#Directory for documents
DOCDIR = ~/public_html/eodocs
#Directory for indices -- not useful for the user
IDXDIR = ~/index

EXTRA_DIST=LICENSE

###############################################################################

lib:	
	pushd src; $(MAKE) all; popd

test:	test/run_tests
	pushd test; $(MAKE) all; ./run_tests; touch run_tests; popd 

# so that make doc always compiles the doc ...
doc:    doc/eo.cfg
	pushd doc; $(MAKE) doc; touch eo.cfg; popd

tutorial:
	pushd tutorial; $(MAKE) all; popd

app:
	pushd app; $(MAKE) all; popd

all:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) all; popd; done

clean:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) clean; popd; done

###############################################################################
