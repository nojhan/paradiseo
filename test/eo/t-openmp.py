#!/usr/bin/env python

#
# (c) Thales group, 2010
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation;
#    version 2 of the License.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Contact: http://eodev.sourceforge.net
#
# Authors:
# Caner Candan <caner.candan@thalesgroup.com>
#

import optparse, logging, sys, os
from datetime import datetime

LEVELS = {'debug': logging.DEBUG,
	  'info': logging.INFO,
	  'warning': logging.WARNING,
	  'error': logging.ERROR,
	  'critical': logging.CRITICAL}

LOG_DEFAULT_FILENAME='notitle.log'

RESULT_FILE_FORMAT='%s%s_p%d_pS%d_P%d_d%d_dS%d_D%d_r%d_s%d'

def parser(parser=optparse.OptionParser()):
    # general parameters
    parser.add_option('-v', '--verbose', choices=LEVELS.keys(), default='info', help='set a verbose level')
    parser.add_option('-f', '--file', help='give an input project filename', default='')
    parser.add_option('-o', '--output', help='give an output filename for logging', default=LOG_DEFAULT_FILENAME)
    # general parameters ends

    parser.add_option('-r', '--nRun', type='int', default=100, help='how many times you would compute each iteration ?')
    parser.add_option('-s', '--seed', type='int', default=1, help='give here a seed value')
    parser.add_option('-n', '--nProc', type='int', default=1, help='give a number of processus, this value is multiplied by the measures bounds')
    parser.add_option('-F', '--fixedBound', type='int', default=1000, help='give the fixed bound value common for all measures')

    topic = str(datetime.today())
    for char in [' ', ':', '-', '.']: topic = topic.replace(char, '_')
    parser.add_option('-t', '--topic', default='openmp_measures_' + topic + '/', help='give here a topic name used to create the folder')

    parser.add_option('-E', '--onlyexecute', action='store_true', default=False, help='used this option if you only want to execute measures without generating images')
    parser.add_option('-X', '--onlyprint', action='store_true', default=False, help='used this option if you only want to generate images without executing measures, dont forget to set the good path in using --topic with a "/" at the end')

    parser.add_option('-C', '--onlyConstTime', action='store_true', default=False, help='only measures constant time problem')
    parser.add_option('-V', '--onlyVarTime', action='store_true', default=False, help='only measures variable time problem')

    parser.add_option('-m', '--measure', action='append', type='int', help='select all measure you want to produce, by default all are produced')

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

if not options.onlyexecute:
    import pylab

def get_boxplot_data( filename ):
    try:
	f = open( filename )
	return [ [ float(value) for value in line.split() ] for line in f.readlines() ]
    except:
	raise ValueError('got an issue during the reading of file %s' % filename)

def do_measure( name, p, ps, P, d, ds, D, r=options.nRun, s=options.seed, v='logging' ):
    OPENMP_EXEC_FORMAT='./test/t-openmp -p=%d --popStep=%d -P=%d -d=%d --dimStep=%d -D=%d -r=%d --seed=%d -v=%s -H=%s -C=%d -V=%d'

    pwd = options.topic + name + '_'
    cmd = OPENMP_EXEC_FORMAT % (p, ps, P, d, ds, D, r, s, v, pwd,
				0 if options.onlyVarTime else 1,
				0 if options.onlyConstTime else 1)
    logging.info( cmd )
    if not options.onlyprint:
	os.system( cmd )

    if not options.onlyexecute:
	def generate( filenames ):
	    for cur in filenames:
		filename = RESULT_FILE_FORMAT % (pwd, cur, p, ps, P, d, ds, D, r, s)
		pylab.boxplot( get_boxplot_data( filename ) )
		nonzero = lambda x: x if x > 0 else 1
		iters = ( nonzero( P - p ) / ps ) * ( nonzero( D - d ) / ds )
		pylab.xlabel('%d iterations from %d,%d to %d,%d' % ( iters, p, d, P, D) )
		pylab.ylabel('%s - %s' % (cur, name))
		pylab.savefig( filename + '.pdf', format='pdf' )
		pylab.savefig( filename + '.png', format='png' )
		pylab.cla()
		pylab.clf()

	if not options.onlyVarTime:
	    generate( ['speedup', 'efficiency', 'dynamicity'] )
	if not options.onlyConstTime:
	    generate( ['variable_speedup', 'variable_efficiency', 'variable_dynamicity'] )

def main():
    if not options.onlyprint:
	logging.info('creates first the new topic repository %s', options.topic)
	os.mkdir( options.topic )

    logging.info('do all tests with r = %d and a common seed value = %d' % (options.nRun, options.seed))

    logging.info('EA in time O(1) and O(n) - speedup measure Sp, Ep and Dp for P & D')

    n = options.nProc
    F = options.fixedBound

    if options.measure is None or 1 in options.measure:
	logging.info('(1) measure for all combinaisons of P n D')
	do_measure( '1', 1*n, 10*n, 101*n, 1*n, 10*n, 101*n )

    if options.measure is None or 2 in options.measure:
	logging.info('(2) measure for P \in [%d, %d[ with D fixed to %d' % (1*n, 101*n, F))
	do_measure( '2', 1*n, 1*n, 101*n, F, 1, F )

    if options.measure is None or 3 in options.measure:
	logging.info('(3) measure for P \in [%d, %d[ with ps = %d and D fixed to %d' % (1*n, 1001*n, 10*n, F))
	do_measure( '3', 1*n, 10*n, 1001*n, F, 1, F )

    if options.measure is None or 4 in options.measure:
	logging.info('(4) measure for D \in [%d, %d[ with P fixed to %d' % (1*n, 101*n, F))
	do_measure( '4', F, 1, F, 1*n, 1*n, 101*n )

    if options.measure is None or 5 in options.measure:
	logging.info('(5) measure for D \in [%d, %d[ with ds = %d and P fixed to %d' % (1*n, 1001*n, 10*n, F))
	do_measure( '5', F, 1, F, 1*n, 10*n, 1001*n )

# when executed, just run main():
if __name__ == '__main__':
    logging.debug('### plotting started ###')

    main()

    logging.debug('### plotting ended ###')
