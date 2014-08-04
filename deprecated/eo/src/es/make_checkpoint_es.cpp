// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_checkpoint_es.cpp
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

/** This file contains ***INSTANCIATED DEFINITIONS*** of checkpoint fns
 * of the library for evolution of ***ES genotypes*** inside EO.
 * It should be included in the file that calls any of the corresponding fns
 * Compiling this file allows one to generate part of the library (i.e. object
 * files that you just need to link with your own main and fitness code).
 *
 * The corresponding ***INSTANCIATED DECLARATIONS*** are contained
 *       in src/es/es.h
 * while the TEMPLATIZED code is define in make_checkpoint.h in the src/do dir
 */

// The templatized code
#include <do/make_checkpoint.h>
// the instanciating EOType(s)
#include <es/eoEsSimple.h>         // one Sigma per individual
#include <es/eoEsStdev.h>          // one sigmal per object variable
#include <es/eoEsFull.h>           // full correlation matrix per indi

/// The following function merely call the templatized do_* functions

// checkpoint
/////////////
eoCheckPoint<eoEsSimple<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<double> >& _eval, eoContinue<eoEsSimple<double> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}
eoCheckPoint<eoEsSimple<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsSimple<eoMinimizingFitness> >& _eval, eoContinue<eoEsSimple<eoMinimizingFitness> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

/////////////
eoCheckPoint<eoEsStdev<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<double> >& _eval, eoContinue<eoEsStdev<double> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}
eoCheckPoint<eoEsStdev<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsStdev<eoMinimizingFitness> >& _eval, eoContinue<eoEsStdev<eoMinimizingFitness> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

/////////////
eoCheckPoint<eoEsFull<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<double> >& _eval, eoContinue<eoEsFull<double> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}
eoCheckPoint<eoEsFull<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoEsFull<eoMinimizingFitness> >& _eval, eoContinue<eoEsFull<eoMinimizingFitness> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}
