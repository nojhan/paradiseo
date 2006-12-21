// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "make_op_FlowShop.h"

// (c) OPAC Team, LIFL, March 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/

#ifndef _make_op_FlowShop_h
#define _make_op_FlowShop_h

// the operators
#include <eoOp.h>
#include <eoGenOp.h>
#include <eoCloneOps.h>
#include <eoOpContainer.h>
// combinations of simple eoOps (eoMonOp and eoQuadOp)
#include <eoProportionalCombinedOp.h>

// definition of crossover
#include "FlowShopOpCrossoverQuad.h"
// definition of mutation
#include "FlowShopOpMutationShift.h"
#include "FlowShopOpMutationExchange.h"

// also need the parser and state includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/*
 * This function builds the operators that will be applied to the eoFlowShop
 * @param eoParameterLoader& _parser to get user parameters
 * @param eoState& _state to store the memory
 */
eoGenOp < FlowShop > &do_make_op (eoParameterLoader & _parser,
				  eoState & _state)
{

  /////////////////////////////
  // Variation operators
  ////////////////////////////

  // the crossover
  ////////////////

  // a first crossover   
  eoQuadOp < FlowShop > *cross = new FlowShopOpCrossoverQuad;
  // store in the state
  _state.storeFunctor (cross);

  // relative rate in the combination
  double cross1Rate = _parser.createParam (1.0, "crossRate",
					   "Relative rate for the only crossover",
					   0,
					   "Variation Operators").value ();
  // creation of the combined operator with this one
  eoPropCombinedQuadOp < FlowShop > *propXover =
    new eoPropCombinedQuadOp < FlowShop > (*cross, cross1Rate);
  // store in the state
  _state.storeFunctor (propXover);


  // the mutation
  ///////////////

  // a first mutation : the shift mutation
  eoMonOp < FlowShop > *mut = new FlowShopOpMutationShift;
  _state.storeFunctor (mut);
  // its relative rate in the combination
  double mut1Rate = _parser.createParam (0.5, "shiftMutRate",
					 "Relative rate for shift mutation",
					 0,
					 "Variation Operators").value ();
  // creation of the combined operator with this one
  eoPropCombinedMonOp < FlowShop > *propMutation =
    new eoPropCombinedMonOp < FlowShop > (*mut, mut1Rate);
  _state.storeFunctor (propMutation);

  // a second mutation : the exchange mutation
  mut = new FlowShopOpMutationExchange;
  _state.storeFunctor (mut);
  // its relative rate in the combination
  double mut2Rate = _parser.createParam (0.5, "exchangeMutRate",
					 "Relative rate for exchange mutation",
					 0,
					 "Variation Operators").value ();
  // addition of this one to the combined operator
  propMutation->add (*mut, mut2Rate);

  // end of crossover and mutation definitions
  ////////////////////////////////////////////

  // First read the individual level parameters
  eoValueParam < double >&pCrossParam =
    _parser.createParam (0.25, "pCross", "Probability of Crossover", 'c',
			 "Variation Operators");
  // minimum check
  if ((pCrossParam.value () < 0) || (pCrossParam.value () > 1))
    throw runtime_error ("Invalid pCross");

  eoValueParam < double >&pMutParam =
    _parser.createParam (0.35, "pMut", "Probability of Mutation", 'm',
			 "Variation Operators");
  // minimum check
  if ((pMutParam.value () < 0) || (pMutParam.value () > 1))
    throw runtime_error ("Invalid pMut");

  // the crossover - with probability pCross
  eoProportionalOp < FlowShop > *propOp = new eoProportionalOp < FlowShop >;
  _state.storeFunctor (propOp);
  eoQuadOp < FlowShop > *ptQuad = new eoQuadCloneOp < FlowShop >;
  _state.storeFunctor (ptQuad);
  propOp->add (*propXover, pCrossParam.value ());	// crossover, with proba pcross
  propOp->add (*ptQuad, 1 - pCrossParam.value ());	// nothing, with proba 1-pcross

  // now the sequential
  eoSequentialOp < FlowShop > *op = new eoSequentialOp < FlowShop >;
  _state.storeFunctor (op);
  op->add (*propOp, 1.0);	// always do combined crossover
  op->add (*propMutation, pMutParam.value ());	// then mutation, with proba pmut

  // return a reference
  return *op;
}

#endif
