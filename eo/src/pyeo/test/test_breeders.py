from maxone import *
import unittest
	
evaluate = EvalFunc()
init = Init(20)
mutate = Mutate()
xover = Crossover()

class TestSGA(unittest.TestCase):
    
    def runtest(self, breed):
	
	pop = eoPop(50, init)
	for indy in pop: evaluate(indy)
	
	newpop = eoPop();

	breed(pop,newpop)
	
	print pop.best()
	for indy in newpop: evaluate(indy)
	print newpop.best()

    def testGeneralBreeder(self):
	
	seq = eoSequentialOp();
	seq.add(xover, 0.7)
	seq.add(mutate, 0.9)

	sel = eoDetTournamentSelect(3)

	breed = eoGeneralBreeder(sel, seq)

	self.runtest(breed)

def suite():
    return unittest.makeSuite(TestSGA,'test')


if __name__=='__main__':
    unittest.main()
