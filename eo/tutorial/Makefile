SUBDIRS = Lesson1 Lesson2 Lesson3 Lesson4

lesson1 : 
	cd Lesson1; make

lesson2 : 
	cd Lesson2; make

lesson3 : 
	cd Lesson3; make

lesson4 : 
	cd Lesson4; make

all:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE); popd; done

clean:
	for i in $(SUBDIRS); do pushd $$i && $(MAKE) clean; popd; done
