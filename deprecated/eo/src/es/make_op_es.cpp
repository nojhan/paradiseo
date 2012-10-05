// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_op_real.cpp
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

/** This file contains ***INSTANCIATED DEFINITIONS*** of operators fns
 * of the library for ***eoReal*** evolution inside EO.
 * It should be included in the file that calls any of the corresponding fns
 * Compiling this file allows one to generate part of the library (i.e. object
 * files that you just need to link with your own main and fitness code).
 *
 * The corresponding ***INSTANCIATED DECLARATIONS*** are contained
 *       in es.h in src/es dir
 * while the TEMPLATIZED code is define in make_op.h in the es dir
 *
 */

// Templatized code
#include <es/make_op_es.h>

/// The following function merely call the templatized do_* functions above

// operators
////////////
eoGenOp<eoEsSimple<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsSimple<double> >& _init)
{
  return do_make_op(_parser, _state, _init);
}

eoGenOp<eoEsSimple<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsSimple<eoMinimizingFitness> >& _init)
{
  return do_make_op(_parser, _state, _init);
}

eoGenOp<eoEsStdev<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsStdev<double> >& _init)
{
  return do_make_op(_parser, _state, _init);
}

eoGenOp<eoEsStdev<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsStdev<eoMinimizingFitness> >& _init)
{
  return do_make_op(_parser, _state, _init);
}

eoGenOp<eoEsFull<double> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsFull<double> >& _init)
{
  return do_make_op(_parser, _state, _init);
}

eoGenOp<eoEsFull<eoMinimizingFitness> >&  make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<eoEsFull<eoMinimizingFitness> >& _init)
{
  return do_make_op(_parser, _state, _init);
}
