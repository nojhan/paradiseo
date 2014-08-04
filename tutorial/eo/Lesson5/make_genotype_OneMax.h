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

#include "eoOneMax.h"
#include "eoOneMaxInit.h"
  // also need the parser and param includes
#include <paradiseo/eo/utils/eoParser.h>
#include <paradiseo/eo/utils/eoState.h>


/*
 * This fuction does the create an eoInit<eoOneMax>
 *
 * It could be here tempatized only on the fitness, as it can be used
 * to evolve structures with any fitness.
 * However, for consistency reasons, it was finally chosen, as in
 * the rest of EO, to templatize by the full EOT, as this eventually
 * allows to choose the type of genotype at run time (see in es dir)
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

template <class EOT>
eoInit<EOT> & do_make_genotype(eoParameterLoader& _parser, eoState& _state, EOT)
{
  // read any useful parameter here from the parser
  // the param itself will belong to the parser (as far as memory is concerned)

  //    paramType & param = _parser.createParam(deafultValue, "Keyword", "Comment to appear in help and status", 'c',"Section of status file").value();

  unsigned vecSize = _parser.createParam(unsigned(8), "VecSize", "Size of the bitstrings", 'v',"Representation").value();

  // Then built the initializer - a pointer, stored in the eoState
  eoInit<EOT>* init = new eoOneMaxInit<EOT>(vecSize);
  // store in state
  _state.storeFunctor(init);
  // and return a reference
  return *init;
}

#endif
