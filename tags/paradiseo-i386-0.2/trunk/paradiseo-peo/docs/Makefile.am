## Makefile.am for paradiseo/docs


EXTRA_DIST 		= paradiseo-peo.doxyfile foot.html

doc:
			cd $(srcdir) && doxygen paradiseo-peo.doxyfile

clean-local:
			cd $(srcdir) && rm -rf html latex man
