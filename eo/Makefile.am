###############################################################################
##
## Makefile.am for eo
##
###############################################################################

SUBDIRS = src test
DOCDIR = ~/public_html/eodocs

docs: 
	doc++ -d $(DOCDIR) -B foot.html -f src/*.h src/*.cpp
	/home/jmerelo/bin/index -e html -i /home/jmerelo/index/neweo.idx /home/jmerelo/public_html/eodocs/ -v 3 