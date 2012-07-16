// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// es.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2001
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

/** This file contains all ***INSTANCIATED*** declarations of all components
 * of the library for ***ES-like gnptype*** evolution inside EO.
 * It should be included in the file that calls any of the corresponding fns
 *
 * The corresponding ***INSTANCIATED*** definitions are contained in
 * the different .cpp files in the src/es dir,
 * while the TEMPLATIZED code is define in the different make_XXX.h files
 * either in hte src/do dir for representation independant functions,
 * or in the src/es dir for representation dependent stuff.
 *
 * See also real.h for the similar declarations of eoReal genotypes
 *   i.e. ***without*** mutation parameters attached to individuals
 *
 * Unlike most EO .h files, it does not (and should not) contain any code,
 * just declarations
 */

#ifndef es_h
#define es_h

#include <eoAlgo.h>
#include <eoScalarFitness.h>
#include <utils/eoParser.h>
#include <eoEvalFuncPtr.h>
#include <eoEvalFuncCounter.h>
#include <utils/eoCheckPoint.h>
#include <eoGenOp.h>
#include <eoPop.h>
#include <utils/eoDistance.h>

#include <es/eoEsSimple.h>         // one Sigma per individual
#include <es/eoEsStdev.h>          // one sigmal per object variable
#include <es/eoEsFull.h>           // full correlation matrix per indi

// include all similar declaration for eoReal - i.e. real-valued genotyes
// without self-adaptation
#include <es/make_real.h>

/** @addtogroup Builders
 * @{
 */

//Representation dependent - rewrite everything anew for each representation
//////////////////////////
// the genotypes
eoRealInitBounded<eoEsSimple<double> > & make_genotype(eoParser& _parser, eoState& _state, eoEsSimple<double> _eo);
eoRealInitBounded<eoEsSimple<eoMinimizingFitness> > & make_genotype(eoParser& _parser, eoState& _state, eoEsSimple<eoMinimizingFitness> _eo);

eoRealInitBounded<eoEsStdev<double> > & make_genotype(eoParser& _parser, eoState& _state, eoEsStdev<double> _eo);
eoRealInitBounded<eoEsStdev<eoMinimizingFitness> > & make_genotype(eoParser& _parser, eoState& _state, eoEsStdev<eoMinimizingFitness> _eo);

eoRealInitBounded<eoEsFull<double> > & make_genotype(eoParser& _parser, eoState& _state, eoEsFull<double> _eo);
eoRealInitBounded<eoEsFull<eoMinimizingFitness> > & make_genotype(eoParser& _parser, eoState& _state, eoEsFull<eoMinimizingFitness> _eo);



// the operators
eoGenOp<eoEsSimple<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsSimple<double> >& _init);
eoGenOp<eoEsSimple<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsSimple<eoMinimizingFitness> >& _init);
eoGenOp<eoEsStdev<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsStdev<double> >& _init);
eoGenOp<eoEsStdev<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsStdev<eoMinimizingFitness> >& _init);
eoGenOp<eoEsFull<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsFull<double> >& _init);
eoGenOp<eoEsFull<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsFull<eoMinimizingFitness> >& _init);

//Representation INdependent
////////////////////////////
// you don't need to modify that part even if you use your own representation

// init pop
eoPop<eoEsSimple<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsSimple<double> >&);
eoPop<eoEsSimple<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsSimple<eoMinimizingFitness> >&);

eoPop<eoEsStdev<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsStdev<double> >&);
eoPop<eoEsStdev<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsStdev<eoMinimizingFitness> >&);

eoPop<eoEsFull<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsFull<double> >&);
eoPop<eoEsFull<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoEsFull<eoMinimizingFitness> >&);

// the continue's
eoContinue<eoEsSimple<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<double> > & _eval);
eoContinue<eoEsSimple<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<eoMinimizingFitness> > & _eval);

eoContinue<eoEsStdev<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<double> > & _eval);
eoContinue<eoEsStdev<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<eoMinimizingFitness> > & _eval);

eoContinue<eoEsFull<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<double> > & _eval);
eoContinue<eoEsFull<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<eoMinimizingFitness> > & _eval);

// the checkpoint
eoCheckPoint<eoEsSimple<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<double> >& _eval, eoContinue<eoEsSimple<double> >& _continue);
eoCheckPoint<eoEsSimple<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<eoMinimizingFitness> >& _eval, eoContinue<eoEsSimple<eoMinimizingFitness> >& _continue);

eoCheckPoint<eoEsStdev<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<double> >& _eval, eoContinue<eoEsStdev<double> >& _continue);
eoCheckPoint<eoEsStdev<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<eoMinimizingFitness> >& _eval, eoContinue<eoEsStdev<eoMinimizingFitness> >& _continue);

eoCheckPoint<eoEsFull<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<double> >& _eval, eoContinue<eoEsFull<double> >& _continue);
eoCheckPoint<eoEsFull<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<eoMinimizingFitness> >& _eval, eoContinue<eoEsFull<eoMinimizingFitness> >& _continue);


// the algo
eoAlgo<eoEsSimple<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsSimple<double> >& _eval, eoContinue<eoEsSimple<double> >& _ccontinue, eoGenOp<eoEsSimple<double> >& _op, eoDistance<eoEsSimple<double> >* _dist = NULL);
eoAlgo<eoEsSimple<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsSimple<eoMinimizingFitness> >& _eval, eoContinue<eoEsSimple<eoMinimizingFitness> >& _ccontinue, eoGenOp<eoEsSimple<eoMinimizingFitness> >& _op, eoDistance<eoEsSimple<eoMinimizingFitness> >* _dist = NULL);

eoAlgo<eoEsStdev<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsStdev<double> >& _eval, eoContinue<eoEsStdev<double> >& _ccontinue, eoGenOp<eoEsStdev<double> >& _op, eoDistance<eoEsStdev<double> >* _dist = NULL);
eoAlgo<eoEsStdev<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsStdev<eoMinimizingFitness> >& _eval, eoContinue<eoEsStdev<eoMinimizingFitness> >& _ccontinue, eoGenOp<eoEsStdev<eoMinimizingFitness> >& _op, eoDistance<eoEsStdev<eoMinimizingFitness> >* _dist = NULL);

eoAlgo<eoEsFull<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsFull<double> >& _eval, eoContinue<eoEsFull<double> >& _ccontinue, eoGenOp<eoEsFull<double> >& _op, eoDistance<eoEsFull<double> >* _dist = NULL);
eoAlgo<eoEsFull<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsFull<eoMinimizingFitness> >& _eval, eoContinue<eoEsFull<eoMinimizingFitness> >& _ccontinue, eoGenOp<eoEsFull<eoMinimizingFitness> >& _op, eoDistance<eoEsFull<eoMinimizingFitness> >* _dist = NULL);

// run
void run_ea(eoAlgo<eoEsSimple<double> >& _ga, eoPop<eoEsSimple<double> >& _pop);
void run_ea(eoAlgo<eoEsSimple<eoMinimizingFitness> >& _ga, eoPop<eoEsSimple<eoMinimizingFitness> >& _pop);

void run_ea(eoAlgo<eoEsStdev<double> >& _ga, eoPop<eoEsStdev<double> >& _pop);
void run_ea(eoAlgo<eoEsStdev<eoMinimizingFitness> >& _ga, eoPop<eoEsStdev<eoMinimizingFitness> >& _pop);

void run_ea(eoAlgo<eoEsFull<double> >& _ga, eoPop<eoEsFull<double> >& _pop);
void run_ea(eoAlgo<eoEsFull<eoMinimizingFitness> >& _ga, eoPop<eoEsFull<eoMinimizingFitness> >& _pop);

// end of parameter input (+ .status + help)
// that one is not templatized, but is here for completeness
void make_help(eoParser & _parser);

/** @} */
/** @} */
#endif
