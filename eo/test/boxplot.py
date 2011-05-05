#!/usr/bin/env python

import pylab
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
	print 'Usage: boxplot.py [Results files, ...]'
	sys.exit()

    for i in range(1, len(sys.argv)):
	pylab.boxplot( [ [ float(value) for value in line.split() ] for line in open( sys.argv[i] ).readlines() ] )

    pylab.xlabel('iterations')
    pylab.show()
