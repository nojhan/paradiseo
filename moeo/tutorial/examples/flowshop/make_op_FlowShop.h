/*
* <make_op_FlowShop.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef MAKE_OP_FLOWSHOP_H_
#define MAKE_OP_FLOWSHOP_H_

#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <eoOp.h>
#include <eoGenOp.h>
#include <eoCloneOps.h>
#include <eoOpContainer.h>
#include <eoProportionalCombinedOp.h>
#include <FlowShopOpCrossoverQuad.h>
#include <FlowShopOpMutationShift.h>
#include <FlowShopOpMutationExchange.h>

/*
 * This function builds the operators that will be applied to the eoFlowShop
 * @param eoParameterLoader& _parser to get user parameters
 * @param eoState& _state to store the memory
 */
eoGenOp<FlowShop> & do_make_op(eoParameterLoader& _parser, eoState& _state)
{

  /////////////////////////////
  // Variation operators
  ////////////////////////////

  // the crossover
  ////////////////

  // a first crossover
  eoQuadOp<FlowShop> *cross = new FlowShopOpCrossoverQuad;
  // store in the state
  _state.storeFunctor(cross);

  // relative rate in the combination
  double cross1Rate = _parser.createParam(1.0, "crossRate", "Relative rate for the only crossover", 0, "Variation Operators").value();
  // creation of the combined operator with this one
  eoPropCombinedQuadOp<FlowShop> *propXover = new eoPropCombinedQuadOp<FlowShop>(*cross, cross1Rate);
  // store in the state
  _state.storeFunctor(propXover);


  // the mutation
  ///////////////

  // a first mutation : the shift mutation
  eoMonOp<FlowShop> *mut = new FlowShopOpMutationShift;
  _state.storeFunctor(mut);
  // its relative rate in the combination
  double mut1Rate = _parser.createParam(0.5, "shiftMutRate", "Relative rate for shift mutation", 0, "Variation Operators").value();
  // creation of the combined operator with this one
  eoPropCombinedMonOp<FlowShop> *propMutation = new eoPropCombinedMonOp<FlowShop>(*mut, mut1Rate);
  _state.storeFunctor(propMutation);

  // a second mutation : the exchange mutation
  mut = new FlowShopOpMutationExchange;
  _state.storeFunctor(mut);
  // its relative rate in the combination
  double mut2Rate = _parser.createParam(0.5, "exchangeMutRate", "Relative rate for exchange mutation", 0, "Variation Operators").value();
  // addition of this one to the combined operator
  propMutation -> add(*mut, mut2Rate);

  // end of crossover and mutation definitions
  ////////////////////////////////////////////

  // First read the individual level parameters
  eoValueParam<double>& pCrossParam = _parser.createParam(0.25, "pCross", "Probability of Crossover", 'c', "Variation Operators" );
  // minimum check
  if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
    throw eoParamException("Invalid pCross");

  eoValueParam<double>& pMutParam = _parser.createParam(0.35, "pMut", "Probability of Mutation", 'm', "Variation Operators" );
  // minimum check
  if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
    throw eoParamException("Invalid pMut");

  // the crossover - with probability pCross
  eoProportionalOp<FlowShop> * propOp = new eoProportionalOp<FlowShop> ;
  _state.storeFunctor(propOp);
  eoQuadOp<FlowShop> *ptQuad = new eoQuadCloneOp<FlowShop>;
  _state.storeFunctor(ptQuad);
  propOp -> add(*propXover, pCrossParam.value()); // crossover, with proba pcross
  propOp -> add(*ptQuad, 1-pCrossParam.value()); // nothing, with proba 1-pcross

  // now the sequential
  eoSequentialOp<FlowShop> *op = new eoSequentialOp<FlowShop>;
  _state.storeFunctor(op);
  op -> add(*propOp, 1.0);	 // always do combined crossover
  op -> add(*propMutation, pMutParam.value()); // then mutation, with proba pmut

  // return a reference
  return *op;
}

#endif /*MAKE_OP_FLOWSHOP_H_*/
