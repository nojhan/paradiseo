## Makefile.am for paradiseo/examples


SUBDIRS = shared lesson1 lesson2

doc:
			mkdir ../docs/html/lsnshared; cd shared && doxygen paradiseo-peo-lsn-shared.doxyfile && cd ..
			for ex in `ls | grep lesson`; do mkdir ../docs/html/$$ex; `cd $$ex && doxygen paradiseo-peo-lsn.doxyfile`; done

clean-local:
			rm -fr ../docs/html/lsnshared; cd shared && rm -fr html latex man && cd ..
			for ex in `ls | grep lesson`; do rm -fr ../docs/html/$$ex; `cd $$ex && rm -fr html latex man`; done
