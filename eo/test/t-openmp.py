#!/usr/bin/env python

import pylab
import optparse, logging, sys, os
from datetime import datetime

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

LOG_DEFAULT_FILENAME='notitle.log'

OPENMP_EXEC_FORMAT='./test/t-openmp -p=%d --popStep=%d -P=%d -d=%d --dimStep=%d -D=%d -r=%d --seed=%d -v=%s -H=%s'

def parser(parser=optparse.OptionParser()):
    # general parameters
    parser.add_option('-v', '--verbose', choices=LEVELS.keys(), default='warning', help='set a verbose level')
    parser.add_option('-f', '--file', help='give an input project filename', default='')
    parser.add_option('-o', '--output', help='give an output filename for logging', default=LOG_DEFAULT_FILENAME)
    # general parameters ends

    parser.add_option('-p', '--popMin', default=1)
    parser.add_option('', '--popStep', default=1)
    parser.add_option('-P', '--popMax', default=100)
    parser.add_option('-d', '--dimMin', default=1)
    parser.add_option('', '--dimStep', default=1)
    parser.add_option('-D', '--dimMax', default=100)
    parser.add_option('-r', '--nRun', default=100)
    parser.add_option('-s', '--seed', default=-1)

    topic = str(datetime.today())
    for char in [' ', ':', '-', '.']: topic = topic.replace(char, '_')
    parser.add_option('-t', '--topic', default='openmp_' + topic + '/')

    options, args = parser.parse_args()

    logger(options.verbose, options.output)

    return options

def logger(level_name, filename=LOG_DEFAULT_FILENAME):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        filename=filename, filemode='a'
        )

    console = logging.StreamHandler()
    console.setLevel(LEVELS.get(level_name, logging.NOTSET))
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)

options = parser()

def execute_openmp( p, ps, P, d, ds, D, r, s, v=options.verbose ):
    cmd = OPENMP_EXEC_FORMAT % (p, ps, P, d, ds, D, r, s, v, options.topic)
    logging.debug( cmd )
    #os.system( cmd )

def main():
    # creates first the new topic repository
    #os.mkdir( options.topic )

    # (1) EA in time O(1)

    # (1.1) speedup measure Sp, Ep for P & D

    # (1.1.1) measure for all combinaisons of P n D
    execute_openmp( 1, 10, 100, 1, 10, 100, 100, options.seed )

    # (1.1.1) measure for all combinaisons of P n D
    execute_openmp( 1, 10, 100, 1, 10, 100, 100, options.seed )


    # pylab.boxplot( [ [ float(value) for value in line.split() ] for line in open( sys.argv[i] ).readlines() ] )

    # pylab.xlabel('iterations')
    # pylab.savefig( sys.argv[ len(sys.argv) - 1 ], format='pdf', transparent=True )

    # (2) EA in time O(1)


# when executed, just run main():
if __name__ == '__main__':
    logging.debug('### plotting started ###')

    main()

    logging.debug('### plotting ended ###')
