// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// real.h
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
 * of the library for ***std::vector<RealValues>*** evolution inside EO.
 * It should be included in the file that calls any of the corresponding fns
 *
 * The corresponding ***INSTANCIATED*** definitions are contained in
 * the different .cpp files in the src/es dir,
 * while the TEMPLATIZED code is define in the different make_XXX.h files
 * either in hte src/do dir for representation independant functions,
 * or in the src/es dir for representation dependent stuff.
 *
 * See also es.h for the similar declarations of ES-like genotypes
 *   i.e. ***with*** mutation parameters attached to individuals
 *
 * Unlike most EO .h files, it does not (and should not) contain any code,
 * just declarations
 */

#ifndef real_h
#define real_h

#include <eoAlgo.h>
#include <eoScalarFitness.h>
#include <utils/eoParser.h>
#include <eoEvalFuncPtr.h>
#include <eoEvalFuncCounter.h>
#include <utils/eoCheckPoint.h>
#include <eoGenOp.h>
#include <eoPop.h>
#include <utils/eoDistance.h>

#include <es/eoRealInitBounded.h>
#include <es/eoReal.h>

//Representation dependent - rewrite everything anew for each representation
//////////////////////////


/** @addtogroup Builders
 * @{
 */
// the genotypes
eoRealInitBounded<eoReal<double> > & make_genotype(eoParser& _parser, eoState& _state, eoReal<double> _eo);
eoRealInitBounded<eoReal<eoMinimizingFitness> > & make_genotype(eoParser& _parser, eoState& _state, eoReal<eoMinimizingFitness> _eo);

// the operators
eoGenOp<eoReal<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoReal<double> >& _init);
eoGenOp<eoReal<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoReal<eoMinimizingFitness> >& _init);

//Representation INdependent
////////////////////////////
// you don't need to modify that part even if you use your own representation

// init pop
eoPop<eoReal<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoReal<double> >&);
eoPop<eoReal<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoReal<eoMinimizingFitness> >&);

// the continue's
eoContinue<eoReal<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoReal<double> > & _eval);
eoContinue<eoReal<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoReal<eoMinimizingFitness> > & _eval);

// the checkpoint
eoCheckPoint<eoReal<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoReal<double> >& _eval, eoContinue<eoReal<double> >& _continue);
eoCheckPoint<eoReal<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoReal<eoMinimizingFitness> >& _eval, eoContinue<eoReal<eoMinimizingFitness> >& _continue);


// the algo
eoAlgo<eoReal<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoReal<double> >& _eval, eoContinue<eoReal<double> >& _ccontinue, eoGenOp<eoReal<double> >& _op, eoDistance<eoReal<double> >* _dist = NULL);

eoAlgo<eoReal<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoReal<eoMinimizingFitness> >& _eval, eoContinue<eoReal<eoMinimizingFitness> >& _ccontinue, eoGenOp<eoReal<eoMinimizingFitness> >& _op, eoDistance<eoReal<eoMinimizingFitness> >* _dist = NULL);

// run
void run_ea(eoAlgo<eoReal<double> >& _ga, eoPop<eoReal<double> >& _pop);
void run_ea(eoAlgo<eoReal<eoMinimizingFitness> >& _ga, eoPop<eoReal<eoMinimizingFitness> >& _pop);

// end of parameter input (+ .status + help)
// that one is not templatized
// Because of that, the source is in src/utils dir
void make_help(eoParser & _parser);

/** @} */
#endif
