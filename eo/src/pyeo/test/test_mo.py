from maxone import *
import unittest

class TestSGA(unittest.TestCase):
    
    def testMO(self):
	setObjectivesSize(2);
	setObjectivesValue(0,1);
	setObjectivesValue(1,-1);

	eo1 = EO();
	eo2 = EO();

	eo1.fitness = [0,1];
	eo2.fitness = [1,1];

	print dominates(eo1, eo2)
	setObjectivesValue(0,-1)
	setObjectivesValue(1,1);
	print dominates(eo1, eo2)

if __name__=='__main__':
    unittest.main()
