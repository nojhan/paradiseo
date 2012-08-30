print 'importing maxone'
from maxone import *
print 'done'
import unittest

class Mut(Mutate):
    def __init__(self):
	Mutate.__init__(self)
	self.cnt = 0;
    def __call__(self, eo):
	self.cnt += 1;
	return Mutate.__call__(self, eo)

class Xover(Crossover):
    def __init__(self):
	Crossover.__init__(self)
	self.cnt = 0;
    def __call__(self, eo1, eo2):
	self.cnt += 1;
	return Crossover.__call__(self, eo1, eo2)

class TestPopulator(unittest.TestCase):
    def make_pop(self):
	pop = eoPop(20, init)
	for indy in pop: evaluate(indy)
	return pop

    def test_sequential(self):
	pop = self.make_pop()
	populator = eoSeqPopulator(pop, pop)

	print populator.get()
	print populator.get()

    def test_selective(self):
	sel = eoDetTournamentSelect(2)
	pop = self.make_pop()

	populator = eoSelectivePopulator(pop, pop, sel)

	print populator.get()
	print populator.get()

    def runOpContainer(self, opcontainer):
	mutate = Mut()
	xover = Xover()

	print 'making seq'
	seq = opcontainer()

	print "xovertype", xover.getType()
	print "mutationtype", mutate.getType()

	seq.add(mutate, 0.4)
	seq.add(xover, 0.8)

	pop = self.make_pop();
	offspring = eoPop()

	sel = eoDetTournamentSelect(2)

	print "making populator"
	populator = eoSelectivePopulator(pop, offspring, sel)
	print 'made'

	for i in xrange(1000):
	    seq(populator)

	print mutate.cnt
	print xover.cnt


    def test_sequentialOp(self):
	print '*'*20, "SequentialOp", '*'*20
	self.runOpContainer(eoSequentialOp)

    def test_proportionalOp(self):
	print '*'*20, "ProportionalOp", '*'*20
	self.runOpContainer(eoProportionalOp)

if __name__=='__main__':
    unittest.main()
