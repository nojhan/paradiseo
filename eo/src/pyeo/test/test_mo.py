from maxone import *
import unittest

class TestSGA(unittest.TestCase):
    
    def testIndividuals(self):
	setObjectivesSize(2);
	setObjectivesValue(0,1);
	setObjectivesValue(1,1);

	eo1 = EO();
	eo2 = EO();

	eo1.fitness = [0,1];
	eo2.fitness = [1,1];

	self.failUnlessEqual(dominates(eo1, eo2), 0)
	self.failUnlessEqual(dominates(eo2, eo1), 1)
	self.failUnlessEqual(dominates(eo2, eo2), 0)

	setObjectivesValue(0,-1)
	setObjectivesValue(1,-1);

	self.failUnlessEqual(dominates(eo1, eo2), 1)
	self.failUnlessEqual(dominates(eo2, eo1), 0)
	self.failUnlessEqual(dominates(eo2, eo2), 0)
   
    def testNDSorting(self):
	
	setObjectivesSize(2);
	setObjectivesValue(0,-1)
	setObjectivesValue(1,-1);
	
	pop = Pop()
	pop.resize(6)

	pop[5].fitness = [0.15,0.87]
	pop[4].fitness = [0.1,0.9]
	pop[3].fitness = [0,1];
	pop[2].fitness = [1,0];
	pop[1].fitness = [1,1];
	pop[0].fitness = [2,1];

	srt = eoNDSorting_II()

	srt(pop)
	srt.sort_pop(pop)

	for indy in pop:
	    print indy.fitness

	
	worths = srt.value()
	print worths
	print type(worths)


if __name__=='__main__':
    unittest.main()
