#!/usr/bin/env python

import pylab
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: boxplot.py [Results files, ...] [output file in .png]'
        sys.exit()

    for i in range(1, len(sys.argv) - 1):
        pylab.boxplot( [ [ float(value) for value in line.split() ] for line in open( sys.argv[i] ).readlines() ] )

    pylab.savefig( sys.argv[ len(sys.argv) - 1 ] )
