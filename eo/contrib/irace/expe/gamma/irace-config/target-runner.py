#!/usr/bin/python
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration ID
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import re
import subprocess
import sys

exe = "../../../../release/onlymutga"

problem = 19
pop_size = 1
offspring_size = 100

fixed_parameters = ["--problem", str(problem), "--crossover-rate", "0", "--mutation-rate", "1", "--pop-size", str(pop_size), " --offspring-size", str(offspring_size)]

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

# Get the parameters as command line arguments.
configuration_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]
slices_prop = sys.argv[5:]
#print(sys.argv)

exe = os.path.expanduser(exe)

cmd = [exe] + fixed_parameters + ["--instance", instance, "--seed", seed]

residual_prob = 1
cl_probs = []
residual_size = 1
cl_sizes = []

values = ""
sizes = ""
for i in range(len(slices_prop)):
    cl_probs.append(residual_prob * float(slices_prop[i]))
    cl_sizes.append(residual_size * (1-float(slices_prop[i])))
    residual_prob -= cl_probs[-1]
    residual_size -= cl_sizes[-1]
    values += "%.2f,"%cl_probs[-1]
    sizes += "%.2f,"%cl_sizes[-1]

cl_probs.append(residual_prob)
values += "%.2f"%cl_probs[-1]
sizes += "%.2f"%cl_sizes[-1]

cmd += ["--cl-probs", values, "--cl-sizes", sizes]


# Define the stdout and stderr files.
out_file = "c" + str(configuration_id) + "-" + str(instance_id) + str(seed) + ".stdout"
err_file = "c" + str(configuration_id) + "-" + str(instance_id) + str(seed) + ".stderr"

def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

def check_executable(fpath):
    fpath = os.path.expanduser(fpath)
    if not os.path.isfile(fpath):
        target_runner_error(str(fpath) + " not found")
    if not os.access(fpath, os.X_OK):
        target_runner_error(str(fpath) + " is not executable")

# This is an example of reading a number from the output.
def parse_output(out):
    match = re.search(r'Best ([-+0-9.eE]+)', out.strip())
    if match:
        return match.group(1);
    else:
        return "No match"
        
check_executable (exe)

outf = open(out_file, "w")
errf = open(err_file, "w")
return_code = subprocess.call(cmd, stdout = outf, stderr = errf)

outf.close()
errf.close()

if return_code != 0:
    target_runner_error("command returned code " + str(return_code))

if not os.path.isfile(out_file):
    target_runner_error("output file " + out_file  + " not found.")

cost = parse_output (open(out_file).read())
#print(cost)
print(open(out_file).read().strip())

os.remove(out_file)
os.remove(err_file)
sys.exit(0)

