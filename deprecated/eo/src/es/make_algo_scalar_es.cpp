// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo_scalar_es.cpp
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

#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

/** This file contains ***INSTANCIATED DEFINITIONS*** of select/replace fns
 * of the library for evolution of ***eoEs genotypes*** inside EO.
 * It should be included in the file that calls any of the corresponding fns
 * Compiling this file allows one to generate part of the library (i.e. object
 * files that you just need to link with your own main and fitness code).
 *
 * The corresponding ***INSTANCIATED DECLARATIONS*** are contained
 *       in src/es/es.h
 * while the TEMPLATIZED code is define in make_algo_scalar.h in the src/do dir
 */

// The templatized code
#include <do/make_algo_scalar.h>
// the instanciating EOType(s)
#include <es/eoEsSimple.h>         // one Sigma per individual
#include <es/eoEsStdev.h>          // one sigmal per object variable
#include <es/eoEsFull.h>           // full correlation matrix per indi

/// The following function merely call the templatized do_* functions above

// Algo
///////
eoAlgo<eoEsSimple<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsSimple<double> >& _eval, eoContinue<eoEsSimple<double> >& _continue, eoGenOp<eoEsSimple<double> >& _op, eoDistance<eoEsSimple<double> >* _dist)
{
    return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoEsSimple<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsSimple<eoMinimizingFitness> >& _eval, eoContinue<eoEsSimple<eoMinimizingFitness> >& _continue, eoGenOp<eoEsSimple<eoMinimizingFitness> >& _op, eoDistance<eoEsSimple<eoMinimizingFitness> >* _dist)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

//////////////
eoAlgo<eoEsStdev<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsStdev<double> >& _eval, eoContinue<eoEsStdev<double> >& _continue, eoGenOp<eoEsStdev<double> >& _op, eoDistance<eoEsStdev<double> >* _dist)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoEsStdev<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsStdev<eoMinimizingFitness> >& _eval, eoContinue<eoEsStdev<eoMinimizingFitness> >& _continue, eoGenOp<eoEsStdev<eoMinimizingFitness> >& _op, eoDistance<eoEsStdev<eoMinimizingFitness> >* _dist)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

///////////////
eoAlgo<eoEsFull<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsFull<double> >& _eval, eoContinue<eoEsFull<double> >& _continue, eoGenOp<eoEsFull<double> >& _op, eoDistance<eoEsFull<double> >* _dist)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoEsFull<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoEsFull<eoMinimizingFitness> >& _eval, eoContinue<eoEsFull<eoMinimizingFitness> >& _continue, eoGenOp<eoEsFull<eoMinimizingFitness> >& _op, eoDistance<eoEsFull<eoMinimizingFitness> >* _dist)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}
