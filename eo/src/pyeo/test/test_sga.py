from maxone import *
import unittest

class TestSGA(unittest.TestCase):
    
    def test(self):
	evaluate = EvalFunc()
	init = Init(20)
	mutate = Mutate()
	xover = Crossover()

	pop = Pop(50, init)
	for indy in pop: evaluate(indy)

	select = eoDetTournamentSelect(3);
	cont = eoGenContinue(20);

	sga = eoSGA(select, xover, 0.6, mutate, 0.4, evaluate, cont);

	sga(pop)

	print pop.best()

if __name__=='__main__':
    unittest.main()
