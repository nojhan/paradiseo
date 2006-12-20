## Makefile.am for paradiseo/docs


EXTRA_DIST 		= paradiseo.cfg foot.html

doc:
			cd $(srcdir) && doxygen paradiseo.cfg

clean-local:
			cd $(srcdir) && rm -rf html latex man
