## Makefile.am for eo

if USE_APPLICATIONS
SUBDIRS_APP		= app
endif

if USE_TUTORIAL
SUBDIRS_TUT		= tutorial
endif

SUBDIRS		= src test doc contrib win $(SUBDIRS_APP) $(SUBDIRS_TUT)


# Directory for documents
DOCDIR 		= ~/public_html/eodocs

#Directory for indices -- not useful for the user
IDXDIR 		= ~/index


dist-hook:
		$(MAKE) documentation
		cp -p -r $(srcdir)/doc/html $(srcdir)/doc/latex $(srcdir)/doc/man $(distdir)/doc/

documentation:
		cd doc && $(MAKE) doc && cd ..

doc: 		documentation
