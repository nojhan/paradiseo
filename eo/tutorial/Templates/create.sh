#! /bin/tcsh -f
if ($#argv < 2) then
    echo Usage $argv[0] ApplicationName TargetDir
    exit
endif
if (! -e $2) then
    mkdir $2
endif
sed s/MyStruct/$1/g eoMyStruct.tmpl > $2/eo$1.h
sed s/MyStruct/$1/g init.tmpl > $2/eo$1Init.h
sed s/MyStruct/$1/g evalFunc.tmpl > $2/eo$1EvalFunc.h
sed s/MyStruct/$1/g mutation.tmpl > $2/eo$1Mutation.h
sed s/MyStruct/$1/g quadCrossover.tmpl > $2/eo$1QuadCrossover.h
sed s/MyStruct/$1/g MyStructEA.cpp > $2/$1EA.cpp
sed s/MyStruct/$1/g make_genotype_MyStruct.h > $2/make_genotype_$1.h
sed s/MyStruct/$1/g make_op_MyStruct.h > $2/make_op_$1.h
sed s/MyStruct/$1/g Makefile.tmpl > $2/Makefile


