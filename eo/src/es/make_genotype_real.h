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

#include <es/eoReal.h>
#include <eoEsChromInit.h>
  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/*
 * This fuction does the initialization of what's needed for a particular 
 * genotype (here, vector<double> == eoReal).
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
eoEsChromInit<EOT> & do_make_genotype(eoParameterLoader& _parser, eoState& _state, EOT)
{
  // the fitness type
  typedef typename EOT::Fitness FitT;
  
  // for eoReal, only thing needed is the size
    eoValueParam<unsigned>& vecSize = _parser.createParam(unsigned(10), "vecSize", "The number of variables ", 'n',"Genotype Initialization");

    // to build an eoReal Initializer, we need bounds
    eoValueParam<eoParamParamType>& boundsParam = _parser.createParam(eoParamParamType("(0,1)"), "initBounds", "Bounds for uniform initialization", 'B', "Genotype Initialization");

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
  eoRealVectorBounds * ptBounds = NULL;
  if (v.size() == 2) // a min and a max for all variables 
      ptBounds = new eoRealVectorBounds(vecSize.value(), v[0], v[1]);
  else				   // no time now
    throw runtime_error("Sorry, only unique bounds for all variables implemented at the moment. Come back later");
  // we need to give ownership of this pointer to somebody

  // now some initial value for sigmas - even if useless?
  // shoudl be used in Normal mutation
    eoValueParam<double>& sigmaParam = _parser.createParam(0.3, "sigmaInit", "Initial value for Sigma(s)", 's',"Genotype Initialization");

    // minimum check
  if ( (sigmaParam.value() < 0) )
    throw runtime_error("Invalid sigma");

  eoEsChromInit<EOT> * init = 
    new eoEsChromInit<EOT>(*ptBounds, sigmaParam.value());
  // satore in state
  _state.storeFunctor(init);
  return *init;
}

#endif
