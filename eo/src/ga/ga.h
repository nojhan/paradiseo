#ifndef ga_h
#define ga_h

#include <eoAlgo.h>
#include <eoScalarFitness.h>
#include <utils/eoParser.h>
#include <eoEvalFunc.h>
#include <utils/eoCheckPoint.h>
#include <eoPop.h>

#include <ga/eoBit.h>
#include <ga/eoBitOp.h>


eoAlgo<eoBit<double> >&  make_ga(eoParameterLoader& _parser, eoEvalFunc<eoBit<double> >& _eval, eoCheckPoint<eoBit<double> >& _checkpoint, eoState& state);
eoAlgo<eoBit<eoMinimizingFitness> >&  make_ga(eoParameterLoader& _parser, eoEvalFunc<eoBit<eoMinimizingFitness> >& _eval, eoCheckPoint<eoBit<eoMinimizingFitness> >& _checkpoint, eoState& state);

eoPop<eoBit<double> >&  init_ga(eoParameterLoader& _parser, eoState& _state, double);
eoPop<eoBit<eoMinimizingFitness> >&  init_ga(eoParameterLoader& _parser, eoState& _state, eoMinimizingFitness);

void run_ga(eoAlgo<eoBit<double> >& _ga, eoPop<eoBit<double> >& _pop);
void run_ga(eoAlgo<eoBit<eoMinimizingFitness> >& _ga, eoPop<eoBit<eoMinimizingFitness> >& _pop);

#endif
