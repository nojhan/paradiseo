SUBDIRS = docs src examples

clean_all: clean_aux clean_doc
	-@make clean
	-@(rm -rf aclocal.m4 autom4te.cache configure config.* CO* dep* INST* install-sh miss* mkins* Makefile Makefile.in;\
	cd docs; rm -rf Makefile Makefile.in; cd ../src ; rm -rf Makefile Makefile.in; cd ../examples; rm -rf Makefile Makefile.in;\
	cd tsp; rm -rf .deps/ Makefile Makefile.in; cd share ; rm -rf .deps/)

clean_aux:
	-@find . \( -name "*~" -o -name "*.old" -o -name "#*" -o -name "*.bak" \) -print -exec rm -rf \{\} \;

doc:
	-@(cd docs; make doc)

clean_doc:
	-@(cd docs; make clean-local)
