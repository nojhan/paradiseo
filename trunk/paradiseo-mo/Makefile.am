SUBDIRS = doc src tutorial

CXX=g++

AM_CXXFLAGS =\
	-Wall\
	-Werror\
	-Wno-deprecated\
	-ansi\
	-pedantic

clean_all: clean_aux clean_doc
	-@make clean
	-@(rm -rf acconfig.h acinclude.m4 aclocal.m4 autom4te.cache copying install news readme stamp* texput.log authors todo;\
	rm -f configure config.* CO* dep* INST* install-sh miss* mkins* Makefile Makefile.in;\
	cd docs; rm -rf Makefile Makefile.in; cd ../src ; rm -rf Makefile Makefile.in; cd ../tutorial; rm -rf Makefile Makefile.in;\
	cd examples; rm -f Makefile Makefile.in; cd  tsp ; rm -f Makefile Makefile.in; cd src ; rm -rf .deps/ Makefile Makefile.in;\
	cd ../../../Lesson1 ; rm -rf .deps/ Makefile Makefile.in; cd ../Lesson2 ; rm -rf .deps/ Makefile Makefile.in ; cd ../Lesson3 ;\
	rm -rf .deps/ Makefile Makefile.in)

clean_aux:
	-@find . \( -name "*~" -o -name "*.old" -o -name "#*" -o -name "*.bak" \) -print -exec rm -rf \{\} \;

doc:
	-@(cd doc; make doc)

clean_doc:
	-@(cd doc; make clean-local)

.PHONY: clean_all clean_aux doc clean_doc
