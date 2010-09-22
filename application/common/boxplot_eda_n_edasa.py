#!/usr/bin/env python

from pylab import *
#from pprint import pprint

FILE_LOCATIONS = 'EDA_ResPop/list_of_files.txt'

data = []

locations = [ line.split()[0] for line in open( FILE_LOCATIONS ) ]
#pprint( locations )

for cur_file in locations:
    fitnesses = [ float(line.split()[0]) for line in open( cur_file ).readlines()[1:-1] ]
    data.append( fitnesses[1:] )

#pprint( data )

boxplot( data )

# FILE_LOCATIONS = 'EDASA_ResPop/list_of_files.txt'

# data = []

# locations = [ line.split()[0] for line in open( FILE_LOCATIONS ) ]
# #pprint( locations )

# for cur_file in locations:
#     fitnesses = [ float(line.split()[0]) for line in open( cur_file ).readlines()[1:-1] ]
#     data.append( fitnesses[1:] )

# #pprint( data )

# boxplot( data )

show()
