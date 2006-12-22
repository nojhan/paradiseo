SUBDIRS = shared lesson1

## Makefile.am for paradiseo/examples


EXTRA_DIST 		= peoExamples.cfg foot.html

doc:
			for ex in `ls lesson*`; do cd $(srcdir) && doxygen peoExamples.cfg; done

clean-local:
			for ex in `ls lesson*`; do cd $(ex) && rm -rf html latex man; done
