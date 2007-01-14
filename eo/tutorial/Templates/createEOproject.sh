#! /bin/tcsh -f
#
# Script to create a completely new EO project from templates
#
# Originally by Marc Schoenauer
# Copyright (C) 2006 Jochen Küpper <jochen@fhi-berlin.mpg.de>

if ($PWD:t != Templates) then
    echo "You must start this shell script from the EO Template directory"
    exit
endif

if ($#argv < 1) then
    echo "Usage: $0 ApplicationName [TargetDirName]"
    echo "    This will create ../TargetDirName if necessary (default dir name = ApplicationName)," 
    echo "    and will also put all the files there that are strictly necessary to compile and run"
    echo "    your application."
    exit
endif

# we're going to do something
echo " "
if ($#argv == 1) then
    set TargetDir = /tmp/$1
else
    set TargetDir = /tmp/$2
endif
if ( -d $TargetDir ) then
    echo "Warning: The target directory does exist already."
    echo -n "Overwrite (yes/no)? "
    set REP = $<
    if ($REP != "yes") then
       echo "Stopping, nothing done!"
       exit
    endif
else if ( -e $TargetDir ) then
    echo "Warning: $TargetDir exist but isn't a directory."
    echo "Stopping, nothing done!"
    exit
endif
mkdir -p $TargetDir/src


# creating files
echo "Creating source files for application $1 in $TargetDir/src"
sed s/MyStruct/$1/g eoMyStruct.tmpl > $TargetDir/src/eo$1.h
sed s/MyStruct/$1/g init.tmpl > $TargetDir/src/eo$1Init.h
sed s/MyStruct/$1/g stat.tmpl > $TargetDir/src/eo$1Stat.h
sed s/MyStruct/$1/g evalFunc.tmpl > $TargetDir/src/eo$1EvalFunc.h
sed s/MyStruct/$1/g mutation.tmpl > $TargetDir/src/eo$1Mutation.h
sed s/MyStruct/$1/g quadCrossover.tmpl > $TargetDir/src/eo$1QuadCrossover.h
sed s/MyStruct/$1/g MyStructSEA.cpp > $TargetDir/src/$1EA.cpp

echo "Creating build-support files for application $1 in $TargetDir"
sed s/MyStruct/$1/g configure.ac.tmpl > $TargetDir/configure.ac
sed s/MyStruct/$1/g Makefile.am.top-tmpl > $TargetDir/Makefile.am
sed s/MyStruct/$1/g Makefile.am.src-tmpl > $TargetDir/src/Makefile.am
sed s/MyStruct/$1/g README.tmpl > $TargetDir/README
touch $TargetDir/AUTHORS
touch $TargetDir/COPYING
touch $TargetDir/ChangeLog
touch $TargetDir/INSTALL
touch $TargetDir/NEWS

echo "Successfully created project $1 in $TargetDir!"
echo "Start building the new project"


# building new project
cd $TargetDir
aclocal  || exit
autoheader  || exit
automake --add-missing --copy --force-missing --gnu  || exit
autoconf  || exit
./configure  || exit
make  || exit


# done
echo ""
echo "Project $1 successfully build in $TargetDir!"
echo "Implement your code and enjoy."
