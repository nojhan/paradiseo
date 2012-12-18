#!/usr/bin/env python

from pprint import *
import sys, os

if __name__ == '__main__':

    # parameter phase

    if len(sys.argv) < 2:
        print 'Usage: %s [FILE]' % sys.argv[0]
        sys.exit()

    filename = sys.argv[1]

    lines = open(filename).readlines()

    # formatting phase

    try:
        results = [ x.split() for x in lines[1:-1] ]
    except IOError, e:
        print 'Error: %s' % e
        sys.exit()

    # dimension estimating phase

    popsize = int(lines[0].split()[0])
    dimsize = int(results[0][1])

    # printing phase

    print 'popsize: %d' % popsize
    print 'dimsize: %d' % dimsize

    print
    pprint( results )

    # cvs converting phase

    i = 1
    for x in results:
        x.insert(0, '"%d"' % i)
        i += 1

    header = ['""', '"fitness"', '"dimsize"']

    for i in range(0, dimsize):
        header.append( '"dim%d"' % i )

    results.insert(0, header)

    # cvs printing phase

    file_results = '\n'.join( [ ','.join( x ) for x in results ] )

    print
    print file_results

    try:
        open('%s.csv' % filename, 'w').write(file_results + '\n')
    except IOError, e:
        print 'Error: %s' % e
        sys.exit()

    # ggobi plotting phase

    os.system('ggobi %s.csv' % filename)
