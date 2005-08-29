// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_genotype.h
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

#ifndef _make_genotype_h
#define _make_genotype_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream>
#endif

#include <iostream>

#include "es/eoReal.h"
#include "es/eoEsChromInit.h"
#include "utils/eoRealVectorBounds.h"
  // also need the parser and param includes
#include "utils/eoParser.h"
#include "utils/eoState.h"


/*
 * This fuction does the initialization of what's needed for a particular
 * genotype (here, std::vector<double> == eoReal).
 * It could be here tempatied only on the fitness, as it can be used to evolve
 * bitstrings with any fitness.
 * However, for consistency reasons, it was finally chosen, as in
 * the rest of EO, to templatize by the full EOT, as this eventually
 * allows to choose the type of genotype at run time (see in es dir)
 *
 * It is instanciated in src/es/make_genotyupe_real.cpp
 * and incorporated in the src/es/libes.a
 *
 * It returns an eoInit<EOT> tha can later be used to initialize
 * the population (see make_pop.h).
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is to disambiguate the call upon different instanciations.
 *
 * WARNING: that last argument will generally be the result of calling
 *          the default ctor of EOT, resulting in most cases in an EOT
 *          that is ***not properly initialized***
*/

template <class EOT>
eoEsChromInit<EOT> & do_make_genotype(eoParser& _parser, eoState& _state, EOT)
{
    // the fitness type
    typedef typename EOT::Fitness FitT;

    // for eoReal, only thing needed is the size - but might have been created elswhere ...
    eoValueParam<unsigned>& vecSize
        = _parser.getORcreateParam(unsigned(10), "vecSize",
                                   "The number of variables ",
                                   'n',"Genotype Initialization");
    // to build an eoReal Initializer, we need bounds: [-1,1] by default
    eoValueParam<eoRealVectorBounds>& boundsParam
        = _parser.getORcreateParam(eoRealVectorBounds(vecSize.value(), -1, 1),
                                   "initBounds",
                                   "Bounds for initialization (MUST be bounded)",
                                   'B', "Genotype Initialization");
    // now some initial value for sigmas - even if useless?
    // shoudl be used in Normal mutation
    std::string& sigmaString
        = _parser.getORcreateParam(std::string("0.3"), "sigmaInit",
                                   "Initial value for Sigmas (with a '%' -> scaled by the range of each variable)",
				   's',"Genotype Initialization").value();
    // check for %
    bool to_scale = false;
    size_t pos =  sigmaString.find('%');
    if (pos < sigmaString.size())
    {
        //  found a % - use scaling and get rid of '%'
	to_scale = true;
	sigmaString.resize(pos);
    }
#ifdef HAVE_SSTREAM
    std::istringstream is(sigmaString);
#else
    std::istrstream is(sigmaString.c_str());
#endif
    double sigma;
    is >> sigma;
    // minimum check
    if ( (sigma < 0) )
        throw std::runtime_error("Negative sigma in make_genotype");
    eoEsChromInit<EOT> * init = new eoEsChromInit<EOT>(boundsParam.value(), sigma, to_scale);
    // store in state
    _state.storeFunctor(init);
    return *init;
}

#endif
