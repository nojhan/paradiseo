from maxone import *
import unittest

class TestReduce(unittest.TestCase):
    def run_test(self, ReduceClass, Arg = None):
	pop = eoPop(10, init)
	for indy in pop: evaluate(indy)

	if Arg:
	    red = ReduceClass(Arg)
	else:
	    red = ReduceClass()

	red(pop, 5);

	self.failUnlessEqual( len(pop), 5)

    def test_eoTruncate(self):
	self.run_test(eoTruncate)
    def test_eoRandomeReduce(self):
	self.run_test(eoRandomReduce)
    def test_eoEPRReduce(self):
	self.run_test(eoEPReduce, 2)
    def test_eoLinearTruncate(self):
	self.run_test(eoLinearTruncate)
    def test_eoDetTournamentTruncate(self):
	self.run_test(eoDetTournamentTruncate, 2)
    def test_eoStochTournamentTruncate(self):
	self.run_test(eoStochTournamentTruncate, 0.9)

if __name__=='__main__':
    unittest.main()
