## Process this file with automake to produce Makefile.in
## Created by Anjuta - will be overwritten
## If you don't want it to overwrite it,
## 	Please disable it in the Anjuta project configuration

SUBDIRS = src examples docs

libparadiseodocdir = ${prefix}/docs/paradisEO
libparadiseodoc_DATA = \
	README\
	COPYING\
	AUTHORS\
	ChangeLog\
	INSTALL\
	NEWS\
	TODO

EXTRA_DIST = $(libparadiseodoc_DATA)

# Add documentation to distribution
dist-hook:
	for specfile in *.spec; do \
		if test -f $$specfile; then \
			cp -p $$specfile $(distdir); \
	fi \
	done
	$(MAKE) documentation
	$(INSTALL_DATA) -d $(srcdir)/docs/html $(srcdir)/docs/latex $(srcdir)/docs/man $(distdir)/docs/

# Prepare documentation
documentation:
	cd docs && $(MAKE) doc && cd .. && cd examples && $(MAKE) doc && cd ..

# convenience-target
doc: documentation

