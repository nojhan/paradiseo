// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_op_MyStruct.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2001
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

#ifndef _make_op_MyStruct_h
#define _make_op_MyStruct_h

// the operators
#include <eoOp.h>
#include <eoGenOp.h>
#include <eoCloneOps.h>
#include <eoOpContainer.h>
// combinations of simple eoOps (eoMonOp and eoQuadOp)
#include <eoProportionalCombinedOp.h>

/** definition of mutation:
 * class eoMyStructMonop MUST derive from eoMonOp<eoMyStruct>
 */
#include "eoMyStructMutation.h"

/** definition of crossover (either as eoBinOp (2->1) or eoQuadOp (2->2):
 * class eoMyStructBinCrossover MUST derive from eoBinOp<eoMyStruct>
 * OR
 * class eoMyStructQuadCrossover MUST derive from eoQuadOp<eoMyStruct>
 */
// #include "eoMyStructBinOp.h"
// OR
#include "eoMyStructQuadCrossover.h"

  // also need the parser and state includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// variation operators ///////////////
// canonical (crossover + mutation) only at the moment //

/*
 * This function builds the operators that will be applied to the eoMyStruct
 *
 * It uses a parser (to get user parameters), a state (to store the memory)
 *    the last parameter is an eoInit: if some operator needs some info
 *    about the genotypes, the init has it all (e.g. bounds, ...)
 *    Simply do
 *        EOT myEO;
 *        _init(myEO);
 *    and myEO is then an ACTUAL object
 *
 * As usual, the template is the complete EOT even though only the fitness
 * is actually templatized here: the following only applies to eoMyStruct
*/

template <class EOT>
eoGenOp<EOT> & do_make_op(eoParameterLoader& _parser, eoState& _state, eoInit<EOT>& _init)
{
  // this is a temporary version, while Maarten codes the full tree-structured
  // general operator input
  // BTW we must leave that simple version available somehow, as it is the one
  // that 90% people use!


    /////////////////////////////
    // Variation operators
    ////////////////////////////
    // read crossover and mutations, combine each in a proportional Op
    // and create the eoGenOp that calls crossover at rate pCross
    // then mutation with rate pMut

    // the crossovers
    /////////////////

    // here we can have eoQuadOp (2->2) only - no time for the eoBinOp case

    // you can have more than one - combined in a proportional way

    // first, define the crossover objects and read their rates from the parser

    // A first crossover
    eoQuadOp<Indi> *cross = new eoMyStructQuadCrossover<Indi> /* (varType  _anyVariable) */;
    // store in the state
    _state.storeFunctor(cross);

  // read its relative rate in the combination
    double cross1Rate = _parser.createParam(1.0, "cross1Rate", "Relative rate for crossover 1", '1', "Variation Operators").value();

  // and create the combined operator with this one
  eoPropCombinedQuadOp<Indi> *propXover =
    new eoPropCombinedQuadOp<Indi>(*cross, cross1Rate);
  // and of course stor it in the state
    _state.storeFunctor(propXover);


    // Optional: A second(and third, and ...)  crossover
    //   of course you must create the corresponding classes
    // and all ***MUST*** derive from eoQuadOp<Indi>

  /* Uncomment if necessary - and replicate as many time as you need
      cross = new eoMyStructSecondCrossover<Indi>(varType  _anyVariable);
      _state.storeFunctor(cross);
      double cross2Rate = _parser.createParam(1.0, "cross2Rate", "Relative rate for crossover 2", '2', "Variation Operators").value();
      propXover.add(*cross, cross2Rate);
  */
  // if you want some gentle output, the last one shoudl be like
  //  propXover.add(*cross, crossXXXRate, true);


  // the mutation: same story
  ////////////////
  // you can have more than one - combined in a proportional way

  // for each mutation,
  // - define the mutator object
  // - read its rate from the parser
  // - add it to the proportional combination

  // a first mutation
  eoMonOp<Indi> *mut = new eoMyStructMutation<Indi>/* (varType  _anyVariable) */;
  _state.storeFunctor(mut);
  // its relative rate in the combination
  double mut1Rate = _parser.createParam(1.0, "mut1Rate", "Relative rate for mutation 1", '1', "Variation Operators").value();
  // and the creation of the combined operator with this one
  eoPropCombinedMonOp<Indi> *propMutation = new eoPropCombinedMonOp<Indi>(*mut, mut1Rate);
  _state.storeFunctor(propMutation);

    // Optional: A second(and third, and ...)  mutation with their rates
    //   of course you must create the corresponding classes
    // and all ***MUST*** derive from eoMonOp<Indi>

  /* Uncomment if necessary - and replicate as many time as you need
      mut = new eoMyStructSecondMutation<Indi>(varType  _anyVariable);
      _state.storeFunctor(mut);
      double mut2Rate = _parser.createParam(1.0, "mut2Rate", "Relative rate for mutation 2", '2', "Variation Operators").value();
       propMutation.add(*mut, mut2Rate);
  */
  // if you want some gentle output, the last one shoudl be like
  //  propMutation.add(*mut, mutXXXRate, true);

  // end of crossover and mutation definitions
  ////////////////////////////////////////////

// END Modify definitions of objects by eventually add parameters
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// from now on, you do not need to modify anything
// though you CAN add things to the checkpointing (see tutorial)

  // now build the eoGenOp:
  // to simulate SGA (crossover with proba pCross + mutation with proba pMut
  // we must construct
  //     a sequential combination of
  //          with proba 1, a proportional combination of
  //                        a QuadCopy and our crossover
  //          with proba pMut, our mutation

  // but of course you're free to use any smart combination you could think of
  // especially, if you have to use eoBinOp rather than eoQuad Op youùll have
  // to modify that part

  // First read the individual level parameters
    eoValueParam<double>& pCrossParam = _parser.createParam(0.6, "pCross", "Probability of Crossover", 'C', "Variation Operators" );
    // minimum check
    if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
      throw runtime_error("Invalid pCross");

    eoValueParam<double>& pMutParam = _parser.createParam(0.1, "pMut", "Probability of Mutation", 'M', "Variation Operators" );
    // minimum check
    if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
      throw runtime_error("Invalid pMut");


  // the crossover - with probability pCross
  eoProportionalOp<Indi> * propOp = new eoProportionalOp<Indi> ;
  _state.storeFunctor(propOp);
  eoQuadOp<Indi> *ptQuad = new eoQuadCloneOp<Indi>;
  _state.storeFunctor(ptQuad);
  propOp->add(*propXover, pCrossParam.value()); // crossover, with proba pcross
  propOp->add(*ptQuad, 1-pCrossParam.value()); // nothing, with proba 1-pcross

  // now the sequential
  eoSequentialOp<Indi> *op = new eoSequentialOp<Indi>;
  _state.storeFunctor(op);
  op->add(*propOp, 1.0);	 // always do combined crossover
  op->add(*propMutation, pMutParam.value()); // then mutation, with proba pmut

  // that's it - return a reference
  return *op;
}
#endif
