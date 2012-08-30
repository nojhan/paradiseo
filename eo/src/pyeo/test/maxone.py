import sys
sys.path.append('..')

print 'importing pyeo'
from libPyEO import *
print 'done'

from copy import copy

class MinimFit(float):
    def __cmp__(self, other):
        if other == None: # I seem to be getting None's, don't know why
            return 1
        return float.__cmp__(other, self)

class EvalFunc(eoEvalFunc):
    def __call__(self, eo):
        eo.fitness = reduce(lambda x,y: x+y, eo.genome, 0)

class MinEvalFunc(eoEvalFunc):
    def __call__(self, eo):
        f = reduce(lambda x,y: x+y, eo.genome, 0 )
        eo.fitness = MinimFit(f)

class Init(eoInit):
    def __init__(self, genome_length = 10):
        eoInit.__init__(self)
        self.length = genome_length
    def __call__(self, eo):
        eo.genome = [rng().flip(0.5) for x in range(self.length)]

class Mutate(eoMonOp):
    def __call__(self, eo):
        eo.genome = copy(eo.genome)

        prob = 1. / len(eo.genome)
        for i in range(len(eo.genome)):
            if rng().flip(0.5):
                eo.genome[i] = 1-eo.genome[i];
        return 1

class Crossover(eoQuadOp):
    def __call__(self, eo1, eo2):
        eo1.genome = copy(eo1.genome);
        eo2.genome = copy(eo2.genome);

        point = rng().random( len(eo1.genome) );

        eo1.genome[:point] = eo2.genome[:point];
        eo2.genome[point:] = eo1.genome[point:];

        return 1

evaluate = EvalFunc()
init = Init(3)
mutate = Mutate()
xover = Crossover()

if __name__ == '__main__':

    eo = EO()
    eo1 = EO()

    init(eo1)
    init(eo)
    evaluate(eo)
    print eo

    for i in range(10):
        xover(eo, eo1)
        mutate(eo)

        evaluate(eo)
        print eo, eo1

    print
    print
    print

    pop = eoPop(1, init)

    pop[0] = eo;

    mutate(pop[0])
    pop[0].invalidate()
    evaluate(pop[0])

    print pop[0], eo
