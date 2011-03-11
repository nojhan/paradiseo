/*
* <moeoIteratedIBMOLS.h>
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

#ifndef MOEOITERATEDIBMOLS_H_
#define MOEOITERATEDIBMOLS_H_

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoOp.h>
#include <eoPop.h>
#include <utils/rnd_generators.h>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moeoIBMOLS.h>
#include <moeoPopLS.h>
#include <archive/moeoArchive.h>
#include <fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <moMoveIncrEval.h>



//#include <rsCrossQuad.h>



/**
 * Iterated version of IBMOLS as described in
 * Basseur M., Burke K. : "Indicator-Based Multi-Objective Local Search" (2007).
 */
template < class MOEOT, class Move >
class moeoIteratedIBMOLS : public moeoPopLS < Move>
  {
  public:

    /** The type of objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor.
     * @param _moveInit the move initializer
     * @param _nextMove the neighborhood explorer
     * @param _eval the full evaluation
     * @param _moveIncrEval the incremental evaluation
     * @param _fitnessAssignment the fitness assignment strategy
     * @param _continuator the stopping criteria
     * @param _arch the archive
     * @param _monOp the monary operator
     * @param _randomMonOp the random monary operator (or random initializer)
     * @param _nNoiseIterations the number of iterations to apply the random noise
     */
    moeoIteratedIBMOLS(
      moMoveInit < Move > & _moveInit,
      moNextMove < Move > & _nextMove,
      eoEvalFunc < MOEOT > & _eval,
      moMoveIncrEval < Move, ObjectiveVector > & _moveIncrEval,
      moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > & _fitnessAssignment,
      eoContinue < MOEOT > & _continuator,
      moeoArchive < MOEOT > & _arch,
      eoMonOp < MOEOT > & _monOp,
      eoMonOp < MOEOT > & _randomMonOp,
      unsigned int _nNoiseIterations=1
    ) :
        ibmols(_moveInit, _nextMove, _eval, _moveIncrEval, _fitnessAssignment, _continuator, _arch),
        eval(_eval),
        continuator(_continuator),
        arch(_arch),
        monOp(_monOp),
        randomMonOp(_randomMonOp),
        nNoiseIterations(_nNoiseIterations)
    {}


    /**
     * Apply the local search iteratively until the stopping criteria is met.
     * @param _pop the initial population
     */
    void operator() (eoPop < MOEOT > & _pop)
    {
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            eval(_pop[i]);
        }

      arch(_pop);
      ibmols(_pop);
      while (continuator(arch))
        {
          // generate new solutions from the archive
          generateNewSolutions(_pop);
          // apply the local search (the global archive is updated in the sub-function)
          ibmols(_pop);
        }
    }


  private:

    /** the local search to iterate */
    moeoIBMOLS < MOEOT, Move > ibmols;
    /** the full evaluation */
    eoEvalFunc < MOEOT > & eval;
    /** the stopping criteria */
    eoContinue < MOEOT > & continuator;
    /** archive */
    moeoArchive < MOEOT > & arch;
    /** the monary operator */
    eoMonOp < MOEOT > & monOp;
    /** the random monary operator (or random initializer) */
    eoMonOp < MOEOT > & randomMonOp;
    /** the number of iterations to apply the random noise */
    unsigned int nNoiseIterations;


    /**
     * Creates new population randomly initialized and/or initialized from the archive _arch.
     * @param _pop the output population
     */
    void generateNewSolutions(eoPop < MOEOT > & _pop)
    {
      // shuffle vector for the random selection of individuals
      std::vector<unsigned int> shuffle;
      shuffle.resize(std::max(_pop.size(), arch.size()));
      // init shuffle
      for (unsigned int i=0; i<shuffle.size(); i++)
        {
          shuffle[i] = i;
        }
      // randomize shuffle
      UF_random_generator <unsigned int> gen;
      std::random_shuffle(shuffle.begin(), shuffle.end(), gen);
      // start the creation of new solutions
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          if (shuffle[i] < arch.size()) // the given archive contains the individual i
            {
              // add it to the resulting pop
              _pop[i] = arch[shuffle[i]];
              // apply noise
              for (unsigned int j=0; j<nNoiseIterations; j++)
                {
                  monOp(_pop[i]);
                }
            }
          else // a random solution needs to be added
            {
              // random initialization
              randomMonOp(_pop[i]);
            }
          // evaluation of the new individual
          _pop[i].invalidate();
          eval(_pop[i]);
        }
    }





///////////////////////////////////////////////////////////////////////////////////////////////////////
// A DEVELOPPER RAPIDEMENT POUR TESTER AVEC CROSSOVER //
    /*
    	void generateNewSolutions2(eoPop < MOEOT > & _pop, const moeoArchive < MOEOT > & _arch)
    	{
    		// here, we must have a QuadOp !
    		//eoQuadOp < MOEOT > quadOp;
    		rsCrossQuad quadOp;
    		// shuffle vector for the random selection of individuals
    		vector<unsigned int> shuffle;
    		shuffle.resize(_arch.size());
    		// init shuffle
    		for (unsigned int i=0; i<shuffle.size(); i++)
    		{
    			shuffle[i] = i;
    		}
    		// randomize shuffle
    		UF_random_generator <unsigned int int> gen;
    		std::random_shuffle(shuffle.begin(), shuffle.end(), gen);
    		// start the creation of new solutions
    		unsigned int i=0;
    		while ((i<_pop.size()-1) && (i<_arch.size()-1))
    		{
    			_pop[i] = _arch[shuffle[i]];
    			_pop[i+1] = _arch[shuffle[i+1]];
    			// then, apply the operator nIterationsNoise times
    			for (unsigned int j=0; j<nNoiseIterations; j++)
    			{
    				quadOp(_pop[i], _pop[i+1]);
    			}
    			eval(_pop[i]);
    			eval(_pop[i+1]);
    			i=i+2;
    		}
    		// do we have to add some random solutions ?
    		while (i<_pop.size())
    		{
    			randomMonOp(_pop[i]);
    			eval(_pop[i]);
    			i++;
    		}
    	}
    	*/
///////////////////////////////////////////////////////////////////////////////////////////////////////





  };

#endif /*MOEOITERATEDIBMOLS_H_*/
