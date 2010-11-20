#!/usr/bin/env python

from pylab import *
import sys

DEFAULT_RESULTS_NAME = 'results.txt'

if __name__ == '__main__':
    boxplot( [ [ float(value) for value in line.split() ] for line in open( DEFAULT_RESULTS_NAME if len(sys.argv) < 2 else sys.argv[1] ).readlines() ] )
    show()
