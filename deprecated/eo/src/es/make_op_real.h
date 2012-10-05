// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_op.h - the real-valued version
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

// the specialized Real stuff
#include <es/eoReal.h>
#include <es/eoEsChromInit.h>
#include <es/eoRealOp.h>
#include <es/eoNormalMutation.h>
  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/** @addtogroup Builders
 * @{
 */

/*
 * This function builds the operators that will be applied to the eoReal
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
*/

template <class EOT>
eoGenOp<EOT> & do_make_op(eoParser& _parser, eoState& _state, eoRealInitBounded<EOT>& _init)
{
  // get std::vector size
  unsigned vecSize = _init.size();

  // First, decide whether the objective variables are bounded
  eoValueParam<eoRealVectorBounds>& boundsParam
      = _parser.getORcreateParam(eoRealVectorBounds(vecSize,eoDummyRealNoBounds), "objectBounds",
                                 "Bounds for variables", 'B', "Variation Operators");

  // this is a temporary version(!),
  // while Maarten codes the full tree-structured general operator input
  // BTW we must leave that simple version available somehow, as it is the one
  // that 90% people use!
  eoValueParam<std::string>& operatorParam
      = _parser.getORcreateParam(std::string("SGA"), "operator",
                                 "Description of the operator (SGA only now)",
                                 'o', "Variation Operators");

  if (operatorParam.value() != std::string("SGA"))
    throw std::runtime_error("Sorry, only SGA-like operator available right now\n");

    // now we read Pcross and Pmut,
    // the relative weights for all crossovers -> proportional choice
    // the relative weights for all mutations -> proportional choice
    // and create the eoGenOp that is exactly
    // crossover with pcross + mutation with pmut

  eoValueParam<double>& pCrossParam
      = _parser.getORcreateParam(0.6, "pCross",
                                 "Probability of Crossover",
                                 'C', "Variation Operators" );
  // minimum check
  if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
    throw std::runtime_error("Invalid pCross");

  eoValueParam<double>& pMutParam
      = _parser.getORcreateParam(0.1, "pMut",
                                 "Probability of Mutation",
                                 'M', "Variation Operators" );
  // minimum check
  if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
    throw std::runtime_error("Invalid pMut");

    // the crossovers
    /////////////////
    // the parameters
  eoValueParam<double>& alphaParam
      = _parser.getORcreateParam(double(0.0), "alpha",
                                 "Bound for factor of linear recombinations",
                                 'a', "Variation Operators" );
  // minimum check
  if ( (alphaParam.value() < 0) )
    throw std::runtime_error("Invalid BLX coefficient alpha");


  eoValueParam<double>& segmentRateParam
      = _parser.getORcreateParam(double(1.0), "segmentRate",
                                 "Relative rate for segment crossover",
                                 's', "Variation Operators" );
  // minimum check
  if ( (segmentRateParam.value() < 0) )
    throw std::runtime_error("Invalid segmentRate");

  eoValueParam<double>& hypercubeRateParam
      = _parser.getORcreateParam(double(1.0), "hypercubeRate",
                                 "Relative rate for hypercube crossover",
                                 'A', "Variation Operators" );
  // minimum check
  if ( (hypercubeRateParam.value() < 0) )
    throw std::runtime_error("Invalid hypercubeRate");

  eoValueParam<double>& uxoverRateParam
      = _parser.getORcreateParam(double(1.0), "uxoverRate",
                                 "Relative rate for uniform crossover",
                                 'A', "Variation Operators" );
  // minimum check
  if ( (uxoverRateParam.value() < 0) )
    throw std::runtime_error("Invalid uxoverRate");

    // minimum check
  bool bCross = true;
  if (segmentRateParam.value()+hypercubeRateParam.value()+uxoverRateParam.value()==0)
    {
      std::cerr << "Warning: no crossover" << std::endl;
      bCross = false;
    }

  // Create the CombinedQuadOp
  eoPropCombinedQuadOp<EOT> *ptCombinedQuadOp = NULL;
  eoQuadOp<EOT> *ptQuad = NULL;

  if (bCross)
    {
      // segment crossover for bitstring - pass it the bounds
      ptQuad = new eoSegmentCrossover<EOT>(boundsParam.value(), alphaParam.value());
      _state.storeFunctor(ptQuad);
      ptCombinedQuadOp = new eoPropCombinedQuadOp<EOT>(*ptQuad, segmentRateParam.value());

        // hypercube crossover
      ptQuad = new eoHypercubeCrossover<EOT>(boundsParam.value(), alphaParam.value());
      _state.storeFunctor(ptQuad);
      ptCombinedQuadOp->add(*ptQuad, hypercubeRateParam.value());

        // uniform crossover
      ptQuad = new eoRealUXover<EOT>();
      _state.storeFunctor(ptQuad);
      ptCombinedQuadOp->add(*ptQuad, uxoverRateParam.value());

      // don't forget to store the CombinedQuadOp
      _state.storeFunctor(ptCombinedQuadOp);
    }

  // the mutations
  /////////////////
  // the parameters
  eoValueParam<double> & epsilonParam
      = _parser.getORcreateParam(0.01, "epsilon",
                                 "Half-size of interval for Uniform Mutation",
                                 'e', "Variation Operators" );
  // minimum check
  if ( (epsilonParam.value() < 0) )
    throw std::runtime_error("Invalid epsilon");

  eoValueParam<double> & uniformMutRateParam
      = _parser.getORcreateParam(1.0, "uniformMutRate",
                                 "Relative rate for uniform mutation",
                                 'u', "Variation Operators" );
  // minimum check
  if ( (uniformMutRateParam.value() < 0) )
    throw std::runtime_error("Invalid uniformMutRate");

  eoValueParam<double> & detMutRateParam
      = _parser.getORcreateParam(1.0, "detMutRate",
                                 "Relative rate for deterministic uniform mutation",
                                 'd', "Variation Operators" );
  // minimum check
  if ( (detMutRateParam.value() < 0) )
    throw std::runtime_error("Invalid detMutRate");

  eoValueParam<double> & normalMutRateParam
      = _parser.getORcreateParam(1.0, "normalMutRate",
                                 "Relative rate for Gaussian mutation", 'd', "Variation Operators" );
  // minimum check
  if ( (normalMutRateParam.value() < 0) )
    throw std::runtime_error("Invalid normalMutRate");

  eoValueParam<double> & sigmaParam
      = _parser.getORcreateParam(0.3, "sigma",
                                 "Sigma (fixed) for Gaussian mutation",
                                 's', "Variation Operators" );

  eoValueParam<double> & pNormalParam
      = _parser.getORcreateParam(1.0, "pNormal",
                                 "Proba. to change each variable for Gaussian mutation",
                                 's', "Variation Operators" );

    // minimum check
  bool bMut = true;
  if (uniformMutRateParam.value()+detMutRateParam.value()+normalMutRateParam.value()==0)
    {
      std::cerr << "Warning: no mutation" << std::endl;
      bMut = false;
    }
  if (!bCross && !bMut)
    throw std::runtime_error("No operator called in SGA operator definition!!!");

    // Create the CombinedMonOp
  eoPropCombinedMonOp<EOT> *ptCombinedMonOp = NULL;
  eoMonOp<EOT> *ptMon = NULL;

  if (bMut)
    {
      // uniform mutation on all components:
      // offspring(i) uniformly chosen in [parent(i)-epsilon, parent(i)+epsilon]
      ptMon = new eoUniformMutation<EOT>(boundsParam.value(), epsilonParam.value());
      _state.storeFunctor(ptMon);
      // create the CombinedMonOp
      ptCombinedMonOp = new eoPropCombinedMonOp<EOT>(*ptMon, uniformMutRateParam.value());

        // mutate exactly 1 component (uniformly) per individual
      ptMon = new eoDetUniformMutation<EOT>(boundsParam.value(), epsilonParam.value());
      _state.storeFunctor(ptMon);
      ptCombinedMonOp->add(*ptMon, detMutRateParam.value());

      // mutate all component using Gaussian mutation
      ptMon = new eoNormalVecMutation<EOT>(boundsParam.value(), sigmaParam.value(), pNormalParam.value());
      _state.storeFunctor(ptMon);
      ptCombinedMonOp->add(*ptMon, normalMutRateParam.value());
      _state.storeFunctor(ptCombinedMonOp);
    }

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
  eoSequentialOp<EOT> & op =  _state.storeFunctor(new eoSequentialOp<EOT>);
  op.add(*cross, 1.0);   // always crossover (but clone with prob 1-pCross
  op.add(*ptCombinedMonOp, pMutParam.value());

  // that's it!
  return op;
}
/** @} */
#endif
