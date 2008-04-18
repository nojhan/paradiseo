## Makefile.am for eo

if USE_APPLICATIONS
SUBDIRS_APP		= app
endif

if USE_TUTORIAL
SUBDIRS_TUT		= tutorial
endif

SUBDIRS			= src doc contrib $(SUBDIRS_APP) $(SUBDIRS_TUT) test


# Directory for documents
DOCDIR 			= ~/public_html/eodocs

# Directory for indices -- not useful for the user
IDXDIR 			= ~/index


# Add documentation to distribution
dist-hook:
			$(MAKE) documentation
			$(INSTALL_DATA) -d $(srcdir)/doc/html $(srcdir)/doc/latex $(srcdir)/doc/man $(distdir)/doc/

# Prepare documentation
documentation:
			cd doc && $(MAKE) doc && cd ..
# convinience-target
doc: 			documentation
