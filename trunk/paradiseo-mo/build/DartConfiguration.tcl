# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: /home/me/Programmation/ProjetEclipse/paradiseo-trunk/paradiseo-mo
BuildDirectory: /home/me/Programmation/ProjetEclipse/paradiseo-trunk/paradiseo-mo/build

# Site is something like machine.domain, i.e. pragmatic.crd
Site: me-laptop

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: Linux-c++

# Submission information
IsCDash: TRUE
DropSite: cdash.inria.fr
DropLocation: /CDash/submit.php?project=ParadisEO
DropSiteUser: 
DropSitePassword: 
DropSiteMode: 
DropMethod: http
TriggerSite: 
ScpCommand: /usr/bin/scp

# Dashboard start time
NightlyStartTime: 00:00:00 EST

# Commands for the build/test/submit cycle
ConfigureCommand: "/usr/bin/cmake" "/home/me/Programmation/ProjetEclipse/paradiseo-trunk/paradiseo-mo"
MakeCommand: /usr/bin/make -i

# CVS options
# Default is "-d -P -A"
CVSCommand: /usr/bin/cvs
CVSUpdateOptions: -d -A -P

# Subversion options
SVNCommand: /usr/bin/svn
SVNUpdateOptions: 

# Generic update command
UpdateCommand: /usr/bin/svn
UpdateOptions: 
UpdateType: svn

# Compiler info
Compiler: /usr/bin/c++

# Dynamic analysis and coverage
PurifyCommand: 
ValgrindCommand: 
ValgrindCommandOptions: 
MemoryCheckCommand: MEMORYCHECK_COMMAND-NOTFOUND
MemoryCheckCommandOptions: 
MemoryCheckSuppressionFile: 
CoverageCommand: /usr/bin/gcov

# Cluster commands
SlurmBatchCommand: SLURM_SBATCH_COMMAND-NOTFOUND
SlurmRunCommand: SLURM_SRUN_COMMAND-NOTFOUND

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summarily terminated.
# Currently set to 25 minutes
TimeOut: 1500

UseLaunchers: 
CurlOptions: 
# warning, if you add new options here that have to do with submit,
# you have to update cmCTestSubmitCommand.cxx

