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

#include <ga/eoBit.h>
#include <eoInit.h>
  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// the bitstring ////////////////
/*
 * This fuction does the initialization of what's needed for a particular
 * genotype (here, bitstrings).
 * It could be here tempatied only on the fitness, as it can be used to evolve
 * bitstrings with any fitness.
 * However, for consistency reasons, it was finally chosen, as in
 * the rest of EO, to templatize by the full EOT, as this eventually
 * allows to choose the type of genotype at run time (see in es dir)
 *
 * It is instanciated in ga/ga.cpp - and incorporated in the ga/libga.a
 *
 * It returns an eoInit<eoBit<FitT> > tha can later be used to initialize
 * the population (see make_pop.h).
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is to disambiguate the call upon different instanciations.
 *
 * WARNING: that last argument will generally be the result of calling
 *          the default ctor of EOT, resulting in most cases in an EOT
 *          that is ***not properly initialized***
 *
 * @ingroup bitstring
 * @ingroup Builders
*/
template <class EOT>
eoInit<EOT> & do_make_genotype(eoParser& _parser, eoState& _state, EOT, float _bias=0.5)
{
  // for bitstring, only thing needed is the size
  // but it might have been already read in the definition fo the performance
  unsigned theSize = _parser.getORcreateParam(unsigned(10), "chromSize", "The length of the bitstrings", 'n',"Problem").value();

  // Then we can built a bitstring random initializer
  // based on boolean_generator class (see utils/rnd_generator.h)
  eoBooleanGenerator * gen = new eoBooleanGenerator(_bias);
  _state.storeFunctor(gen);
  eoInitFixedLength<EOT>* init = new eoInitFixedLength<EOT>(theSize, *gen);
  // store in state
  _state.storeFunctor(init);
  return *init;
}

#endif
