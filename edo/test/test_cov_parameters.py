#!/usr/bin/env python

PSIZE = 10000
MEAN = 0
CMD = "./test/t-edoEstimatorNormalMulti -P=%s -m=%.1f -1=%.1f -2=%.1f -3=%.1f && ./gplot.py -r TestResPop -p -w 5 -u -g %s -G results_for_test_cov_parameters -f %s_gen1"

from os import system
from numpy import arange

if __name__ == '__main__':

    for p1 in list(arange(0.1, 1.1, 0.1)):
	for p2 in list(arange(-1., 0., 0.1)) + list(arange(0., 1.1, 0.1)):
	    for p3 in list(arange(0.1, 1.1, 0.1)):
		gen = '%d_%.1f_%.1f_%.1f_%.1f' % (PSIZE, MEAN, p1, p2, p3)
		cmd = CMD % ( PSIZE, MEAN, p1, p2, p3, gen, gen )
		print cmd
		system( cmd )
