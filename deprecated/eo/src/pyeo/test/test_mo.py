from maxone import *
from math import exp
import unittest

class MyInit(eoInit):
    def __call__(self, eo):
	eo.genome = [rng().normal(), rng().normal(), rng().normal()];

class MyMutate(eoMonOp):
    def __call__(self, eo):
	std = 0.05
	eo.genome = copy(eo.genome)

	eo.genome[0] += rng().normal() * std
	eo.genome[1] += rng().normal() * std
	eo.genome[2] += rng().normal() * std
	return 1

class AnEval(eoEvalFunc):
    def __init__(self):
	eoEvalFunc.__init__(self)

	setObjectivesSize(2);
	setObjectivesValue(0,1);
	setObjectivesValue(1,1);

    def __call__(self, eo):
	x = abs(eo.genome[0])
	y = abs(eo.genome[1])
	z = abs(eo.genome[2])

	eo.fitness = [ x / (x+y+z), y /(x+y+z) ]

import Gnuplot

g = Gnuplot.Gnuplot()
g.reset()

def do_plot(pop):
    l1 = []
    l2 = []

    for indy in pop:
	l1.append(indy.fitness[0])
	l2.append(indy.fitness[1])

    d = Gnuplot.Data(l1,l2, with = 'points')

    d2 = Gnuplot.Data([0,1],[1,0], with='lines')

    g.plot(d,d2)

class NSGA_II(eoAlgo):
    def __init__(self, ngens):
	self.cont = eoGenContinue(ngens);

	self.selectOne = eoDetTournamentSelect(2);
	self.evaluate = AnEval()
	self.mutate = MyMutate()
	self.init = MyInit()

	self.seq = eoProportionalOp()
	self.seq.add(self.mutate, 1.0)

	self.perf2worth = eoNDSorting_II()

    def __call__(self, pop):
	sz = len(pop)
	i = 0
	while self.cont(pop):
	    newpop = eoPop()
	    populator = eoSelectivePopulator(pop, newpop, self.selectOne);

	    while len(newpop) < sz:
		self.seq(populator)

	    for indy in newpop:
		self.evaluate(indy)
		pop.push_back(indy)

	    self.perf2worth(pop)
	    self.perf2worth.sort_pop(pop)

	    #print pop[0].fitness, pop[0].genome
	    pop.resize(sz)

	    #worth = self.perf2worth.getValue()
	    #print worth[0], worth[sz-1]

	    i += 1
	    if i%100 == 0:
		pass
		do_plot(pop)

	worths = self.perf2worth.getValue()

	w0 = int(worths[0]-0.001)

	for i in range(len(pop)):
	    if worths[i] <= w0:
		break;

	    print pop[i].genome
	    print pop[i].fitness

class TestNSGA_II(unittest.TestCase):
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

	pop = eoPop()
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

	worths = srt.getValue()
	print worths
	print type(worths)

    def testNSGA_II(self):
	evaluate = AnEval();
	pop = eoPop(25, MyInit())
	for indy in pop: evaluate(indy)

	nsga = NSGA_II(50)

	nsga(pop)

if __name__=='__main__':
    unittest.main()
