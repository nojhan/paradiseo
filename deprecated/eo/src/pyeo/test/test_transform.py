"""Test script for the eoSGATranform class"""

from copy import deepcopy
from libPyEO import *
from maxone import *

pop = eoPop()

for i in range(10):
        eo = EO()
        init(eo)
        evaluate(eo)
        pop.push_back(eo)

transform = eoSGATransform(xover, 0.8, mutate, 0.2)

def test1(pop, transform):
        pop = deepcopy(pop)
        print "test 1"
        print "Initial population:"
        print pop

        transform(pop)

        print "GM pop:"
        print pop

def test2(pop, transform):
        pop = deepcopy(pop)

        print "test 2"
        print "Initial population"
        print pop

        checkpoint = eoCheckPoint(eoGenContinue(50))
        select = eoSelectNumber(eoDetTournamentSelect(3), 10)
        replace = eoGenerationalReplacement()

        algo = eoEasyEA(checkpoint, evaluate, select, transform, replace)
        algo(pop)

        print "Evoluated pop:"
        print pop

if __name__ == "__main__":
        try:
                test1(pop, transform)
        except:
                import sys
                print
                print "Caught an exception:"
                print sys.exc_type, sys.exc_value
                print

        try:
                test2(pop, transform)
        except:
                import sys
                print
                print "Caught an exception:"
                print sys.exc_type, sys.exc_value
                print
