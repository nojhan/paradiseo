#! /bin/tcsh -f
if ($#argv < 2) then
    echo Usage $argv[0] ApplicationName TargetDir
    exit
endif
sed s/eoMyStruct/eo$1/g eoMyStruct.tmpl > $2/eo$1.h
sed s/eoMyStruct/eo$1/g init.tmpl > $2/eo$1Init.h
sed s/eoMyStruct/eo$1/g evalFunc.tmpl > $2/eo$1EvalFunc.h
sed s/eoMyStruct/eo$1/g mutation.tmpl > $2/eo$1Mutation.h
sed s/eoMyStruct/eo$1/g quadCrossover.tmpl > $2/eo$1QuadCrossover.h
sed s/eoMyStruct/eo$1/g eoMyStructEA.cpp > $2/eo$1EA.cpp
sed s/eoMyStruct/eo$1/g Makefile.tmpl > $2/Makefile

