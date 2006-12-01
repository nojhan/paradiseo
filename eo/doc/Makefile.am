## Makefile.am for eo/doc


EXTRA_DIST 		= EO_EA2001.pdf LeCreusot.pdf eo.cfg index.h \
			mainpage.html publications.html sflogo-hammer1.jpg

doc:
			cd $(srcdir) && $(DOXYGEN) eo.cfg

clean-local:
			cd $(srcdir) && rm -rf html latex man
