// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_pop.h
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

#ifndef _make_pop_h
#define _make_pop_h

#include <ctime>  // for time(0) for random seeding
#include <eoPop.h>
#include <eoInit.h>
#include <utils/eoRNG.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>

/** @defgroup Builders Automatic builders
 *
 * Automatic builders are functions that automagically builds most commons instances for you.
 *
 * All the options you needs are set in the command-line parser.
 * Those functions all start with the "do_make_" prefix.
 *
 * @ingroup Utilities
 */

/**
 * Templatized version of parser-based construct of the population
 * + other initializations that are NOT representation-dependent.
 *
 * It must then be instantiated, and compiled on its own for a given EOType
 * (see e.g. ga.h and ga.pp in dir ga)
 *
 * @ingroup Builders
*/
template <class EOT>
eoPop<EOT>&  do_make_pop(eoParser & _parser, eoState& _state, eoInit<EOT> & _init)
{
  // random seed
    eoValueParam<uint32_t>& seedParam = _parser.getORcreateParam(uint32_t(0), "seed", "Random number seed", 'S');
    if (seedParam.value() == 0)
        seedParam.value() = time(0);
    eoValueParam<unsigned>& popSize = _parser.getORcreateParam(unsigned(20), "popSize", "Population Size", 'P', "Evolution Engine");

  // Either load or initialize
  // create an empty pop and let the state handle the memory
  eoPop<EOT>& pop = _state.takeOwnership(eoPop<EOT>());

  eoValueParam<std::string>& loadNameParam = _parser.getORcreateParam(std::string(""), "Load","A save file to restart from",'L', "Persistence" );
  eoValueParam<bool> & recomputeFitnessParam = _parser.getORcreateParam(false, "recomputeFitness", "Recompute the fitness after re-loading the pop.?", 'r',  "Persistence" );

  if (loadNameParam.value() != "") // something to load
    {
      // create another state for reading
      eoState inState;		// a state for loading - WITHOUT the parser
      // register the rng and the pop in the state, so they can be loaded,
      // and the present run will be the exact continuation of the saved run
      // eventually with different parameters
      inState.registerObject(pop);
      inState.registerObject(rng);
      inState.load(loadNameParam.value()); //  load the pop and the rng
      // the fitness is read in the file:
      // do only evaluate the pop if the fitness has changed
      if (recomputeFitnessParam.value())
        {
          for (unsigned i=0; i<pop.size(); i++)
            pop[i].invalidate();
        }
      if (pop.size() < popSize.value())
        std::cerr << "WARNING, only " << pop.size() << " individuals read in file " << loadNameParam.value() << "\nThe remaining " << popSize.value() - pop.size() << " will be randomly drawn" << std::endl;
      if (pop.size() > popSize.value())
        {
          std::cerr << "WARNING, Load file contained too many individuals. Only the best will be retained" << std::endl;
          pop.resize(popSize.value());
        }
    }
  else				// nothing loaded from a file
    {
      rng.reseed(seedParam.value());
    }

  if (pop.size() < popSize.value()) // missing some guys
    {
      // Init pop from the randomizer: need to use the append function
      pop.append(popSize.value(), _init);
    }

  // for future stateSave, register the algorithm into the state
  _state.registerObject(_parser);
  _state.registerObject(pop);
  _state.registerObject(rng);

  return pop;
}

#endif
