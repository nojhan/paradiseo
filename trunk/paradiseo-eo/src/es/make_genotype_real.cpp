// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_genotype_real.cpp
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

/** This file contains ***INSTANCIATED DEFINITIONS*** of eoReal Init fns
 * It should be included in the file that calls any of the corresponding fns
 * Compiling this file allows one to generate part of the library (i.e. object
 * files that you just need to link with your own main and fitness code).
 *
 * The corresponding ***INSTANCIATED DECLARATIONS*** are contained
 *       in src/es/make_real.h
 * while the TEMPLATIZED code is define in make_genotype_real.h
 *
 * It is instanciated in src/es/make_genotype_real.cpp -
 * and incorporated in the ga/libga.a
 *
 * It returns an eoInit<EOT> that can later be used to initialize
 * the population (see make_pop.h).
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is to disambiguate the call upon different instanciations.
 *
 * WARNING: that last argument will generally be the result of calling
 *          the default ctor of EOT, resulting in most cases in an EOT
 *          that is ***not properly initialized***
*/

// the templatized code
#include <es/make_genotype_real.h>

/// The following functions merely call the templatized do_* functions
eoRealInitBounded<eoReal<double> > & make_genotype(eoParser& _parser,
                                                   eoState& _state,
                                                   eoReal<double> _eo)
{
    return do_make_genotype(_parser, _state, _eo);
}



eoRealInitBounded<eoReal<eoMinimizingFitness> > & make_genotype(eoParser& _parser,
                                                                eoState& _state,
                                                                eoReal<eoMinimizingFitness> _eo)
{
    return do_make_genotype(_parser, _state, _eo);
}
