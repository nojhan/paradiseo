#!/usr/bin/env python

import pylab
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: boxplot_to_png.py [Results files, ...] [output file in .png]'
        sys.exit()

    for i in range(1, len(sys.argv) - 1):
        pylab.boxplot( [ [ float(value) for value in line.split() ] for line in open( sys.argv[i] ).readlines() ] )

    pylab.xlabel('iterations')
    pylab.savefig( sys.argv[ len(sys.argv) - 1 ], format='png', transparent=True, papertype='a0' )
