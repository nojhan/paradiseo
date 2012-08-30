from maxone import *
import unittest

class TestBreeders(unittest.TestCase):
    def runtest(self, breed):
	pop = eoPop(50, Init(20))
	evaluate = EvalFunc()
	print 'HERE'
	for indy in pop: evaluate(indy)
	newpop = eoPop();

	breed(pop,newpop)

	print pop.best()
	for indy in newpop: evaluate(indy)
	print newpop.best()

    def testGeneralBreeder(self):
	seq = eoSequentialOp();
	seq.add(Crossover(), 0.7)
	seq.add(Mutate(), 0.1)

	breed = eoGeneralBreeder(eoDetTournamentSelect(3), seq)
	self.runtest(breed)

def suite():
    return unittest.makeSuite(TestSGA,'test')

if __name__=='__main__':
    unittest.main()
