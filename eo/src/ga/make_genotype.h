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
 * This fuciont does the initialization of what's needed for a particular 
 * genotype (here, bitstrings).
 * It is templatized ***olny on the fitness*** so it can be used to evolve 
 * bitstrings with any fitness.
 * It is instanciated in ga/ga.cpp - and incorporated in the ga/libga.a
 *
 * It returns an eoInit<eoBit<FitT> > tha can later be used to initialize 
 * the population (see make_pop.h).
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is to disambiguate the call upon different instanciations.
*/

template <class FitT>
eoInit<eoBit<FitT> > & do_make_genotype(eoParameterLoader& _parser, eoState& _state, FitT)
{
  // for bitstring, only thing needed is the size
    eoValueParam<unsigned>& chromSize = _parser.createParam(unsigned(10), "ChromSize", "The length of the bitstrings", 'n',"initialization");

  // Then we can built a bitstring random initializer
  // based on boolean_generator class (see utils/rnd_generator.h)
  eoBooleanGenerator * gen = new eoBooleanGenerator;
  _state.storeFunctor(gen);
  eoInitFixedLength<eoBit<FitT> >* init = new eoInitFixedLength<eoBit<FitT> >(chromSize.value(), *gen);
  // satore in state
  _state.storeFunctor(init);
  return *init;
}

#endif
