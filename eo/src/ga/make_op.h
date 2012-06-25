// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_op.h
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

#ifndef _make_op_h
#define _make_op_h

// the operators
#include <eoOp.h>
#include <eoGenOp.h>
#include <eoCloneOps.h>
#include <eoOpContainer.h>
// combinations of simple eoOps (eoMonOp and eoQuadOp)
#include <eoProportionalCombinedOp.h>

// the specialized GA stuff
#include <ga/eoBit.h>
#include <ga/eoBitOp.h>
  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// bitstring operators ///////////////
// canonical (crossover + mutation) only at the moment //

/*
 * This function builds the operators that will be applied to the bitstrings
 *
 * It uses a parser (to get user parameters) and a state (to store the memory)
 * the last argument is an individual, needed for 2 reasons
 *     it disambiguates the call after instanciations
 *     some operator might need some private information about the indis
 *
 * This is why the template is the complete EOT even though only the fitness
 * is actually templatized here: the following only applies to bitstrings
 *
 * Note : the last parameter is an eoInit: if some operator needs some info
 *        about the gneotypes, the init has it all (e.g. bounds, ...)
 *        Simply do
 *        EOT myEO;
 *        _init(myEO);
 *        and myEO is then an ACTUAL object
 *
 * @ingroup bitstring
 * @ingroup Builders
*/

template <class EOT>
eoGenOp<EOT> & do_make_op(eoParser& _parser, eoState& _state, eoInit<EOT>& _init)
{
  // this is a temporary version, while Maarten codes the full tree-structured
  // general operator input
  // BTW we must leave that simple version available somehow, as it is the one
  // that 90% people use!
    eoValueParam<std::string>& operatorParam =  _parser.createParam(std::string("SGA"), "operator", "Description of the operator (SGA only now)", 'o', "Variation Operators");

    if (operatorParam.value() != std::string("SGA"))
        throw std::runtime_error("Only SGA-like operator available right now\n");

    // now we read Pcross and Pmut,
    // the relative weights for all crossovers -> proportional choice
    // the relative weights for all mutations -> proportional choice
    // and create the eoGenOp that is exactly
    // crossover with pcross + mutation with pmut

    eoValueParam<double>& pCrossParam = _parser.createParam(0.6, "pCross", "Probability of Crossover", 'C', "Variation Operators" );
    // minimum check
    if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
      throw std::runtime_error("Invalid pCross");

    eoValueParam<double>& pMutParam = _parser.createParam(0.1, "pMut", "Probability of Mutation", 'M', "Variation Operators" );
    // minimum check
    if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
      throw std::runtime_error("Invalid pMut");

    // the crossovers
    /////////////////
    // the parameters
    eoValueParam<double>& onePointRateParam = _parser.createParam(double(1.0), "onePointRate", "Relative rate for one point crossover", '1', "Variation Operators" );
    // minimum check
    if ( (onePointRateParam.value() < 0) )
      throw std::runtime_error("Invalid onePointRate");

    eoValueParam<double>& twoPointsRateParam = _parser.createParam(double(1.0), "twoPointRate", "Relative rate for two point crossover", '2', "Variation Operators" );
    // minimum check
    if ( (twoPointsRateParam.value() < 0) )
      throw std::runtime_error("Invalid twoPointsRate");

    eoValueParam<double>& uRateParam = _parser.createParam(double(2.0), "uRate", "Relative rate for uniform crossover", 'U', "Variation Operators" );
    // minimum check
    if ( (uRateParam.value() < 0) )
      throw std::runtime_error("Invalid uRate");

    // minimum check
    // bool bCross = true; // not used ?
    if (onePointRateParam.value()+twoPointsRateParam.value()+uRateParam.value()==0)
      {
        std::cerr << "Warning: no crossover" << std::endl;
        // bCross = false;
      }

    // Create the CombinedQuadOp
    eoPropCombinedQuadOp<EOT> *ptCombinedQuadOp = NULL;
    eoQuadOp<EOT> *ptQuad = NULL;
    // 1-point crossover for bitstring
    ptQuad = new eo1PtBitXover<EOT>;
    _state.storeFunctor(ptQuad);
    ptCombinedQuadOp = new eoPropCombinedQuadOp<EOT>(*ptQuad, onePointRateParam.value());

    // uniform crossover for bitstring
    ptQuad = new eoUBitXover<EOT>;
    _state.storeFunctor(ptQuad);
    ptCombinedQuadOp->add(*ptQuad, uRateParam.value());

    // 2-points xover
    ptQuad = new eoNPtsBitXover<EOT>;
    _state.storeFunctor(ptQuad);
    ptCombinedQuadOp->add(*ptQuad, twoPointsRateParam.value());

    // don't forget to store the CombinedQuadOp
    _state.storeFunctor(ptCombinedQuadOp);

    // the mutations
    /////////////////
    // the parameters
    eoValueParam<double> & pMutPerBitParam = _parser.createParam(0.01, "pMutPerBit", "Probability of flipping 1 bit in bit-flip mutation", 'b', "Variation Operators" );
    // minimum check
    if ( (pMutPerBitParam.value() < 0) || (pMutPerBitParam.value() > 0.5) )
      throw std::runtime_error("Invalid pMutPerBit");

    eoValueParam<double> & bitFlipRateParam = _parser.createParam(0.01, "bitFlipRate", "Relative rate for bit-flip mutation", 's', "Variation Operators" );
    // minimum check
    if ( (bitFlipRateParam.value() < 0) )
      throw std::runtime_error("Invalid bitFlipRate");

    // oneBitFlip
    eoValueParam<double> & oneBitRateParam = _parser.createParam(0.01, "oneBitRate", "Relative rate for deterministic bit-flip mutation", 'd', "Variation Operators" );
    // minimum check
    if ( (oneBitRateParam.value() < 0) )
      throw std::runtime_error("Invalid oneBitRate");

    // kBitFlip
    eoValueParam<unsigned> & kBitParam = _parser.createParam((unsigned)1, "kBit", "Number of bit for deterministic k bit-flip mutation", 0, "Variation Operators" );
    // minimum check
    if ( ! kBitParam.value() )
      throw std::runtime_error("Invalid kBit");

    eoValueParam<double> & kBitRateParam = _parser.createParam(0.0, "kBitRate", "Relative rate for deterministic k bit-flip mutation", 0, "Variation Operators" );
    // minimum check
    if ( (kBitRateParam.value() < 0) )
      throw std::runtime_error("Invalid kBitRate");

    // minimum check
    // bool bMut = true; // not used ?
    if (bitFlipRateParam.value()+oneBitRateParam.value()==0)
      {
        std::cerr << "Warning: no mutation" << std::endl;
        // bMut = false;
      }

    // Create the CombinedMonOp
    eoPropCombinedMonOp<EOT> *ptCombinedMonOp = NULL;
    eoMonOp<EOT> *ptMon = NULL;

  // standard bit-flip mutation for bitstring
  ptMon = new eoBitMutation<EOT>(pMutPerBitParam.value());
  _state.storeFunctor(ptMon);
  // create the CombinedMonOp
  ptCombinedMonOp = new eoPropCombinedMonOp<EOT>(*ptMon, bitFlipRateParam.value());

  // mutate exactly 1 bit per individual
  ptMon = new eoDetBitFlip<EOT>;
  _state.storeFunctor(ptMon);
  ptCombinedMonOp->add(*ptMon, oneBitRateParam.value());

  // mutate exactly k bit per individual
  ptMon = new eoDetBitFlip<EOT>(kBitParam.value());
  _state.storeFunctor(ptMon);
  ptCombinedMonOp->add(*ptMon, kBitRateParam.value());

  _state.storeFunctor(ptCombinedMonOp);

  // now build the eoGenOp:
  // to simulate SGA (crossover with proba pCross + mutation with proba pMut
  // we must construct
  //     a sequential combination of
  //          with proba 1, a proportional combination of
  //                        a QuadCopy and our crossover
  //          with proba pMut, our mutation

  // the crossover - with probability pCross
  eoProportionalOp<EOT> * cross = new eoProportionalOp<EOT> ;
  _state.storeFunctor(cross);
  ptQuad = new eoQuadCloneOp<EOT>;
  _state.storeFunctor(ptQuad);
  cross->add(*ptCombinedQuadOp, pCrossParam.value()); // user crossover
  cross->add(*ptQuad, 1-pCrossParam.value()); // clone operator

  // now the sequential
  eoSequentialOp<EOT> *op = new eoSequentialOp<EOT>;
  _state.storeFunctor(op);
  op->add(*cross, 1.0);  // always crossover (but clone with prob 1-pCross
  op->add(*ptCombinedMonOp, pMutParam.value());

  // that's it!
  return *op;
}
#endif
