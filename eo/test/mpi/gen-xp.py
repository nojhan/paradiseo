# -*- coding:utf-8 -*-
"""
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
"""
import json

# Where will be saved the experiments?
EXPERIMENTS_FILENAME = "experiments.json"
# What will be the pattern for experiments filenames?
FILENAME_PATTERN = "%(prefix)s_%(distrib_name)s_%(size)s_%(packet_size)s_%(run)s.txt"

def input_number_at_least( min ):
    n = min - 1
    while n < min:
        try:
            n = int(raw_input("Enter a number greater or equal to %s: "% min))
        except Exception:
            print "Please enter an integer."
    return n

def input_number_between( min, max ):
    n = min - 1
    while n < min or n > max:
        try:
            n = int(raw_input("Enter a number between %s and %s: " % (min,max)))
        except Exception:
            print "Please enter a number."
    return n

def choose_continue():
    print """Do you want to continue?
0. No
1. Yes"""
    return bool( input_number_between(0,1) )

def choose_distribution_uniform():
    distribution = {}
    distribution["name"] = "uniform"
    print "Enter the minimum value (in milliseconds): "
    min = input_number_at_least( 0 )
    distribution["min"] = str(min)
    print "Enter the maximum value (in milliseconds): "
    distribution["max"] = str(input_number_at_least( min ))
    return distribution

def choose_distribution_normal():
    distribution = {}
    distribution["name"] = "normal"
    print "Enter the mean (in milliseconds): "
    distribution["mean"] = str(input_number_at_least( 0 ))
    print "Enter the standard deviation (in milliseconds): "
    distribution["stddev"] = str(input_number_at_least( 0 ))
    return distribution

def choose_distribution_power():
    distribution = {}
    distribution["name"] = "exponential"
    print "Enter the mean (in milliseconds): "
    distribution["mean"] = str(input_number_at_least( 0 ))
    return distribution

def choose_distribution():
    print """Choose your distribution:
0. Uniform
1. Normal
2. Exponential"""
    choice = input_number_between( 0, 2 )
    choose_distrib_functions = [ choose_distribution_uniform, choose_distribution_normal, choose_distribution_power ]
    return choose_distrib_functions[ choice ]()

def choose_packet_size():
    print "Enter the size of a packet (group of elements):"
    return str(input_number_at_least( 0 ))

def choose_size():
    print "Enter the total size (size of population):"
    return str(input_number_at_least( 0 ))

def choose_worker_print():
    print """Should the workers print the time they sleep on stdout?
0. No
1. Yes"""
    return str(input_number_between( 0, 1 ))

def choose_nb_runs():
    print """How many runs should be launched for this configuration? Seeds will be automatically affected to the number
of run+1 (for instance, the first run has a seed of 1, the second has a seed of 2, etc.)."""
    return input_number_at_least( 1 )

def choose_prefix():
    print """What is the name of the experiment? It will be used as the prefix of file names."""
    return raw_input("Enter the prefix name: ")

def main():

    prefix = choose_prefix()
    exps = []

    while True:
        exp = {}
        exp["distribution"] = choose_distribution()

        exp["size"] = choose_size()
        exp["packet_size"] = choose_packet_size()
        exp["worker_print_waiting_time"] = choose_worker_print()
        runs = choose_nb_runs()
        for i in range( runs ):
            exp["seed"] = str(i+1)

            filename_map = exp.copy()
            filename_map["run"] = exp["seed"]
            filename_map["distrib_name"] = exp["distribution"]["name"]
            filename_map["prefix"] = prefix
            filename = FILENAME_PATTERN % filename_map

            exp["filename"] = filename
            copy = exp.copy()
            exps.append( copy )

        if not choose_continue():
            break

    # Write the experiments in a file
    f = file( EXPERIMENTS_FILENAME , 'wb')
    f.write("""{"experiments":[""")
    i = 0
    for exp in exps:
        if i > 0:
            f.write(",\n")
        i += 1
        f.write( json.dumps(exp) )
    f.write("]}")
    f.close()

if __name__ == "__main__":
    main()

