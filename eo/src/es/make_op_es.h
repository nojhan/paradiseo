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
#include <es/eoRealAtomXover.h>
#include <es/eoEsChromInit.h>
#include <es/eoEsMutationInit.h>
#include <es/eoEsMutate.h>
#include <es/eoEsGlobalXover.h>
#include <es/eoEsLocalXover.h>
  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


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
eoGenOp<EOT> & do_make_op(eoParameterLoader& _parser, eoState& _state, eoRealInitBounded<EOT>& _init)
{
  // First, decide whether the objective variables are bounded
  eoValueParam<eoParamParamType>& boundsParam = _parser.createParam(eoParamParamType("(0,1)"), "objectBounds", "Bounds for variables (unbounded if absent)", 'B', "Variation Operators");

  // get vector size
  unsigned vecSize = _init.size();

  // the bounds pointer
  eoRealVectorBounds * ptBounds;
  if (_parser.isItThere(boundsParam))	// otherwise, no bounds
    {
      /////Warning: this code should probably be replaced by creating 
      /////    some eoValueParam<eoRealVectorBounds> with specific implementation
      ////     in eoParser.cpp. At the moment, it is there (cf also make_genotype
      eoParamParamType & ppBounds = boundsParam.value(); // pair<string,vector<string> >
      // transform into a vector<double>
      vector<double> v;
      vector<string>::iterator it;
      for (it=ppBounds.second.begin(); it<ppBounds.second.end(); it++)
	{
	  istrstream is(it->c_str());
	  double r;
	  is >> r;
	  v.push_back(r);
	}
      // now create the eoRealVectorBounds object
      if (v.size() == 2) // a min and a max for all variables 
	ptBounds = new eoRealVectorBounds(vecSize, v[0], v[1]);
      else				   // no time now
	throw runtime_error("Sorry, only unique bounds for all variables implemented at the moment. Come back later");
      // we need to give ownership of this pointer to somebody
      /////////// end of temporary code
    }
  else			   // no param for bounds was given
    ptBounds = new eoRealVectorNoBounds(vecSize); // DON'T USE eoDummyVectorNoBounds
				   // as it does not have any dimension

    // now we read Pcross and Pmut, 
  eoValueParam<string>& operatorParam =  _parser.createParam(string("SGA"), "operator", "Description of the operator (SGA only now)", 'o', "Variation Operators");

  if (operatorParam.value() != string("SGA"))
    throw runtime_error("Sorry, only SGA-like operator available right now\n");

    // now we read Pcross and Pmut, 
    // and create the eoGenOp that is exactly 
    // crossover with pcross + mutation with pmut

  eoValueParam<double>& pCrossParam = _parser.createParam(1.0, "pCross", "Probability of Crossover", 'C', "Variation Operators" );
  // minimum check
  if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
    throw runtime_error("Invalid pCross");

  eoValueParam<double>& pMutParam = _parser.createParam(1.0, "pMut", "Probability of Mutation", 'M', "Variation Operators" );
  // minimum check
  if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
    throw runtime_error("Invalid pMut");


  // crossover
  /////////////
  // ES crossover
  eoValueParam<string>& crossTypeParam = _parser.createParam(string("Global"), "crossType", "Type of ES recombination (gloabl or local)", 'C', "Variation Operators");
  
  eoValueParam<string>& crossObjParam = _parser.createParam(string("Discrete"), "crossObj", "Recombination of object variables (Discrete or Intermediate)", 'O', "Variation Operators");
  eoValueParam<string>& crossStdevParam = _parser.createParam(string("Intermediate"), "crossStdev", "Recombination of mutation strategy parameters (Intermediate or Discrete)", 'S', "Variation Operators");

  // The pointers: first the atom Xover
  eoBinOp<double> *ptObjAtomCross = NULL;
  eoBinOp<double> *ptStdevAtomCross = NULL;
  // then the global one
  eoGenOp<EOT> *ptCross;

  // check for the atom Xovers
  if (crossObjParam.value() == string("Discrete"))
    ptObjAtomCross = new eoRealAtomExchange;
  else if (crossObjParam.value() == string("Intermediate"))
    ptObjAtomCross = new eoRealAtomExchange;
  else throw runtime_error("Invalid Object variable crossover type");

  if (crossStdevParam.value() == string("Discrete"))
    ptStdevAtomCross = new eoRealAtomExchange;
  else if (crossStdevParam.value() == string("Intermediate"))
    ptStdevAtomCross = new eoRealAtomExchange;
  else throw runtime_error("Invalid mutation strategy parameter crossover type");

  // and build the indi Xover 
  if (crossTypeParam.value() == string("Global"))
    ptCross = new eoEsGlobalXover<EOT>(*ptObjAtomCross, *ptStdevAtomCross);
  else if (crossTypeParam.value() == string("Local"))
    ptCross = new eoEsLocalXover<EOT>(*ptObjAtomCross, *ptStdevAtomCross);
  else throw runtime_error("Invalide Object variable crossover type");

  // now that everything is OK, DON'T FORGET TO STORE MEMORY
  _state.storeFunctor(ptObjAtomCross);
  _state.storeFunctor(ptStdevAtomCross);
  _state.storeFunctor(ptCross);

  //  mutation 
  /////////////

  // Ok, time to set up the self-adaptive mutation
  // Proxy for the mutation parameters
  eoEsMutationInit mutateInit(_parser, "Variation Operators");
  
  eoEsMutate<EOT> * ptMon = new eoEsMutate<EOT>(mutateInit, *ptBounds);   
  _state.storeFunctor(ptMon);

  // encapsulate into an eoGenop
  eoMonGenOp<EOT> * op = new eoMonGenOp<EOT>(*ptMon);
  _state.storeFunctor(op);

  // that's it!
  return *op;
}
#endif
