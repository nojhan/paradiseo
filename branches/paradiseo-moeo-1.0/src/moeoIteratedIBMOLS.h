// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIteratedIBMOLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOITERATEDIBMOLS_H_
#define MOEOITERATEDIBMOLS_H_

#include <eoContinue.h>
#include <eoOp.h>
#include <eoPop.h>
#include <utils/rnd_generators.h>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moeoMoveIncrEval.h>
#include <moeoArchive.h>
#include <moeoEvalFunc.h>
#include <moeoLS.h>
#include <moeoIndicatorBasedFitnessAssignment.h>
#include <moeoIndicatorBasedLS.h>



#include <rsCrossQuad.h>



/**
 * Iterated version of IBMOLS as described in
 * Basseur M., Burke K. : "Indicator-Based Multi-Objective Local Search" (2007).
 */
template < class MOEOT, class Move >
class moeoIteratedIBMOLS : public moeoLS < MOEOT, eoPop < MOEOT > & >
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
	 * @param _monOp the monary operator
	 * @param _randomMonOp the random monary operator (or random initializer)
	 * @param _nNoiseIterations the number of iterations to apply the random noise
	 */
	moeoIteratedIBMOLS(
		moMoveInit < Move > & _moveInit,
		moNextMove < Move > & _nextMove,
		moeoEvalFunc < MOEOT > & _eval,
		moeoMoveIncrEval < Move > & _moveIncrEval,
		moeoIndicatorBasedFitnessAssignment < MOEOT > & _fitnessAssignment,
		eoContinue < MOEOT > & _continuator,
		eoMonOp < MOEOT > & _monOp,
		eoMonOp < MOEOT > & _randomMonOp,
		unsigned _nNoiseIterations=1
		) :
	  ibmols(_moveInit, _nextMove, _eval, _moveIncrEval, _fitnessAssignment, _continuator),
		eval(_eval),
		continuator(_continuator),
		monOp(_monOp),
		randomMonOp(_randomMonOp),
		nNoiseIterations(_nNoiseIterations)
	{}


	/**
	 * Apply the local search iteratively until the stopping criteria is met.
	 * @param _pop the initial population
	 * @param _arch the (updated) archive
	 */
	void operator() (eoPop < MOEOT > & _pop, moeoArchive < MOEOT > & _arch)
	{

		_arch.update(_pop);
cout << endl << endl << "***** IBMOLS 1" << endl;
unsigned counter = 2;
		ibmols(_pop, _arch);
		while (continuator(_arch))
		{
			// generate new solutions from the archive
			generateNewSolutions(_pop, _arch);
cout << endl << endl << "***** IBMOLS " << counter++ << endl;
			// apply the local search (the global archive is updated in the sub-function)
			ibmols(_pop, _arch);
		}

	}


private:

	/** the stopping criteria */
	eoContinue < MOEOT > & continuator;
	/** the local search to iterate */
	moeoIndicatorBasedLS < MOEOT, Move > ibmols;
	/** the full evaluation */
	moeoEvalFunc < MOEOT > & eval;
	/** the monary operator */
	eoMonOp < MOEOT > & monOp;
	/** the random monary operator (or random initializer) */
	eoMonOp < MOEOT > & randomMonOp;
	/** the number of iterations to apply the random noise */
	unsigned nNoiseIterations;


	/**
	 * Creates new population randomly initialized and/or initialized from the archive _arch.
	 * @param _pop the output population
	 * @param _arch the archive
	 */
	void generateNewSolutions(eoPop < MOEOT > & _pop, const moeoArchive < MOEOT > & _arch)
	{
		// shuffle vector for the random selection of individuals
		vector<unsigned> shuffle;
		shuffle.resize(std::max(_pop.size(), _arch.size()));
		// init shuffle
		for (unsigned i=0; i<shuffle.size(); i++)
		{
			shuffle[i] = i;
		}
		// randomize shuffle
		UF_random_generator <unsigned int> gen;
		std::random_shuffle(shuffle.begin(), shuffle.end(), gen);
		// start the creation of new solutions
		for (unsigned i=0; i<_pop.size(); i++)
		{
			if (shuffle[i] < _arch.size())
			// the given archive contains the individual i
			{
				// add it to the resulting pop
				_pop[i] = _arch[shuffle[i]];
				// then, apply the operator nIterationsNoise times
				for (unsigned j=0; j<nNoiseIterations; j++)
				{
					monOp(_pop[i]);
				}
			}
			else
			// a randomly generated solution needs to be added
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
	void generateNewSolutions2(eoPop < MOEOT > & _pop, const moeoArchive < MOEOT > & _arch)
	{
		// here, we must have a QuadOp !
		//eoQuadOp < MOEOT > quadOp;
		rsCrossQuad quadOp;
		// shuffle vector for the random selection of individuals
		vector<unsigned> shuffle;
		shuffle.resize(_arch.size());
		// init shuffle
		for (unsigned i=0; i<shuffle.size(); i++)
		{
			shuffle[i] = i;
		}
		// randomize shuffle
		UF_random_generator <unsigned int> gen;
		std::random_shuffle(shuffle.begin(), shuffle.end(), gen);
		// start the creation of new solutions
		unsigned i=0;
		while ((i<_pop.size()-1) && (i<_arch.size()-1))
		{
			_pop[i] = _arch[shuffle[i]];
			_pop[i+1] = _arch[shuffle[i+1]];
			// then, apply the operator nIterationsNoise times
			for (unsigned j=0; j<nNoiseIterations; j++)
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
///////////////////////////////////////////////////////////////////////////////////////////////////////





};

#endif /*MOEOITERATEDIBMOLS_H_*/
