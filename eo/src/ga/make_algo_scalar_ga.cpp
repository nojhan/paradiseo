// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_scalar_ga.cpp
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

/** This file contains all ***INSTANCIATED DEFINITIONS*** of operators
 * of the library for ***BISTRING*** evolution inside EO.
 * It should be included in the file that calls any of the corresponding fns
 * Compiling this file allows one to generate part of the library (i.e. object
 * files that you just need to link with your own main and fitness code).
 *
 * The corresponding ***INSTANCIATED DECLARATIONS*** are contained 
 *       in make_algo_scalar_ga.h
 * while the TEMPLATIZED code is define in make_algo_scalar.h in the do dir
 *
 * Unlike most EO .h files, it does not (and should not) contain any code, 
 * just declarations
 */

// The templatized code
#include <do/make_algo_scalar.h>
// the instanciating EOType
#include <ga/eoBit.h>

/// The following function merely call the templatized do_* functions above

// Algo
///////
eoAlgo<eoBit<double> >&  make_algo_scalar(eoParameterLoader& _parser, eoState& _state, eoEvalFunc<eoBit<double> >& _eval, eoContinue<eoBit<double> >& _continue, eoGenOp<eoBit<double> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoBit<eoMinimizingFitness> >&  make_algo_scalar(eoParameterLoader& _parser, eoState& _state, eoEvalFunc<eoBit<eoMinimizingFitness> >& _eval, eoContinue<eoBit<eoMinimizingFitness> >& _continue, eoGenOp<eoBit<eoMinimizingFitness> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

