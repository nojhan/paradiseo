## Makefile.am for eo/doc


EXTRA_DIST 		= eo.cfg foot.html

doc:
			cd $(srcdir) && $(DOXYGEN) eo.cfg

clean-local:
			cd $(srcdir) && rm -rf html latex man
