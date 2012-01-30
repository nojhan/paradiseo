from maxone import *
import unittest

class TestSGA(unittest.TestCase):
    def dotestSGA(self, evaluate):
        init = Init(20)
        mutate = Mutate()
        xover = Crossover()

        pop = eoPop(50, init)
        for indy in pop: evaluate(indy)

        select = eoDetTournamentSelect(3);
        cont1 = eoGenContinue(20);

        cont = eoCheckPoint(cont1)

        mon = eoGnuplot1DMonitor()

        avg = eoAverageStat()
        bst = eoBestFitnessStat()
        mon.add(avg)
        mon.add(bst)

        # add it to the checkpoint
        cont.add(avg)
        #cont.add(mon)
        cont.add(bst)

        sga = eoSGA(select, xover, 0.6, mutate, 0.4, evaluate, cont);

        sga(pop)

        print pop.best()

    def testSGA_Max(self):
        evaluate = EvalFunc()
        self.dotestSGA(evaluate)

    def testSGA_Min(self):
        evaluate = MinEvalFunc()
        self.dotestSGA(evaluate)

if __name__=='__main__':
    unittest.main()
