SUBDIRS = Lesson1 Lesson2 Lesson3

all:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) all; popd; done

lesson1 : 
	cd Lesson1; make

lesson2 : 
	cd Lesson2; make

lesson3 : 
	cd Lesson3; make

clean:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) clean; popd; done
