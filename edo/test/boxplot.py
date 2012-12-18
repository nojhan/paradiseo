#!/usr/bin/env python

from pylab import *

FILE_LOCATIONS = 'means_distances_results/files_description.txt'

data = []

locations = [ line.split()[0] for line in open( FILE_LOCATIONS ) ]

for cur_file in locations:
    data.append( [ float(line.split()[7]) for line in open( cur_file ).readlines() ] )

print locations
#print data

boxplot( data )

show()
