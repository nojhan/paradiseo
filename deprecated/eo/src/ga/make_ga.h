// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// ga.h
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
 * of the library for ***BISTRING*** evolution inside EO.
 * It should be included in the file that calls any of the corresponding fns
 *
 * The corresponding ***INSTANCIATED*** definitions are contained in ga.cpp
 * while the TEMPLATIZED code is define in the different makeXXX.h files
 *
 * Unlike most EO .h files, it does not (and should not) contain any code,
 * just declarations
 */

#ifndef ga_h
#define ga_h

#include <eoAlgo.h>
#include <eoScalarFitness.h>
#include <utils/eoParser.h>
#include <eoEvalFuncCounter.h>
#include <utils/eoCheckPoint.h>
#include <eoGenOp.h>
#include <eoPop.h>
#include <utils/eoDistance.h>

#include <ga/eoBit.h>

//Representation dependent - rewrite everything anew for each representation
//////////////////////////

/** @addtogroup Builders
 * @{
 */

// the genotypes
eoInit<eoBit<double> > & make_genotype(eoParser& _parser, eoState& _state, eoBit<double> _eo, float _bias=0.5);
  eoInit<eoBit<eoMinimizingFitness> > & make_genotype(eoParser& _parser, eoState& _state, eoBit<eoMinimizingFitness> _eo, float _bias=0.5);

// the operators
eoGenOp<eoBit<double> >&  make_op(eoParser& _parser, eoState& _state, eoInit<eoBit<double> >& _init);
eoGenOp<eoBit<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoInit<eoBit<eoMinimizingFitness> >& _init);

//Representation INdependent
////////////////////////////
// if you use your own representation, simply change the types of templates

// init pop
eoPop<eoBit<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoBit<double> >&);
eoPop<eoBit<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoBit<eoMinimizingFitness> >&);

// the continue's
eoContinue<eoBit<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoBit<double> > & _eval);
eoContinue<eoBit<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoBit<eoMinimizingFitness> > & _eval);

// the checkpoint
eoCheckPoint<eoBit<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoBit<double> >& _eval, eoContinue<eoBit<double> >& _continue);
eoCheckPoint<eoBit<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoBit<eoMinimizingFitness> >& _eval, eoContinue<eoBit<eoMinimizingFitness> >& _continue);


// the algo
eoAlgo<eoBit<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoBit<double> >& _eval, eoContinue<eoBit<double> >& _ccontinue, eoGenOp<eoBit<double> >& _op, eoDistance<eoBit<double> >* _dist = NULL);

eoAlgo<eoBit<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoBit<eoMinimizingFitness> >& _eval, eoContinue<eoBit<eoMinimizingFitness> >& _ccontinue, eoGenOp<eoBit<eoMinimizingFitness> >& _op, eoDistance<eoBit<eoMinimizingFitness> >* _dist = NULL);

// run
void run_ea(eoAlgo<eoBit<double> >& _ga, eoPop<eoBit<double> >& _pop);
void run_ea(eoAlgo<eoBit<eoMinimizingFitness> >& _ga, eoPop<eoBit<eoMinimizingFitness> >& _pop);

// end of parameter input (+ .status + help)
// that one is not templatized
// Because of that, the source is in src/utils dir
void make_help(eoParser & _parser);

/** @} */
#endif
