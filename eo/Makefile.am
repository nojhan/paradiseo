###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test win
#Directory for documents
DOCDIR = ~/public_html/eodocs
#Directory for indices -- not useful for the user
IDXDIR = ~/index

EXTRA_DIST=LICENSE

docs: 
	doc++ -d $(DOCDIR) -B foot.html -f src/*.h src/*.cpp
	/home/jmerelo/bin/index -e html -i $(IDXDIR)/neweo.idx $(DOCDIR)/ -v 3 