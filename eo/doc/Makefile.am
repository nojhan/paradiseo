###############################################################################
##
## Makefile.am for eo/doc
##
###############################################################################


EXTRA_DIST 		= eo.cfg foot.html \
			$(srcdir)/html/* \
			$(srcdir)/latex/* \
			$(srcdir)/man/*


doc:
			cd $(srcdir) && $(DOXYGEN) eo.cfg && touch eo.cfg

clean-local:
			cd $(srcdir) && rm -rf html latex man && touch eo.cfg
