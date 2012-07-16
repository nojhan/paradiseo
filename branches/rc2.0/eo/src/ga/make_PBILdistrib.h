// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_PBILdistrib.h
// (c) Marc Schoenauer, Maarten Keijzer, 2001
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

    Contact: Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _make_PBILdistrib_h
#define _make_PBILdistrib_h

#include <ctime>                   // for time(0) for random seeding
#include <ga/eoPBILOrg.h>
#include <utils/eoRNG.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>


//////////////////////////DISTRIB CONSTRUCTION ///////////////////////////////
/**
 * Templatized version of parser-based construct of the distribution
 * for PBIL distribution evolution algorithm
 *
 * It must then be instantiated, and compiled on its own for a given EOType
 * (see test/t-eoPBIL.cpp
 *
 * Last argument is template-disambiguating
*/


template <class EOT>
eoPBILDistrib<EOT> &  do_make_PBILdistrib(eoParser & _parser, eoState& _state, EOT)
{
  // First the random seed
    eoValueParam<uint32_t>& seedParam = _parser.createParam(uint32_t(0), "seed", "Random number seed", 'S');
    if (seedParam.value() == 0)
        seedParam.value() = time(0);

    // chromosome size:
    unsigned theSize;
    // but it might have been already read in the definition fo the performance
    eoParam* ptParam = _parser.getParamWithLongName(std::string("chromSize"));

    if (!ptParam)                          // not already defined: read it here
      {
        theSize = _parser.createParam(unsigned(10), "chromSize", "The length of the bitstrings", 'n',"Problem").value();
      }
    else                                   // it was read before, get its value
      {
        eoValueParam<unsigned>* ptChromSize = dynamic_cast<eoValueParam<unsigned>*>(ptParam);
        theSize = ptChromSize->value();
      }

    eoPBILDistrib<EOT> * ptDistrib = new eoPBILDistrib<EOT>(theSize);
    _state.storeFunctor(ptDistrib);

    // now the initialization: read a previously saved distribution, or random
  eoValueParam<std::string>& loadNameParam = _parser.createParam(std::string(""), "Load","A save file to restart from",'L', "Persistence" );
  if (loadNameParam.value() != "") // something to load
    {
      // create another state for reading
      eoState inState;		// a state for loading - WITHOUT the parser
      // register the rng and the distribution in the state,
      // so they can be loaded,
      // and the present run will be the exact continuation of the saved run
      // eventually with different parameters
      inState.registerObject(*ptDistrib);
      inState.registerObject(rng);
      inState.load(loadNameParam.value()); //  load the distrib and the rng
    }
  else				// nothing loaded from a file
    {
      rng.reseed(seedParam.value());
    }

  // for future stateSave, register the algorithm into the state
   _state.registerObject(_parser);
   _state.registerObject(*ptDistrib);
   _state.registerObject(rng);

  return *ptDistrib;
}

#endif
