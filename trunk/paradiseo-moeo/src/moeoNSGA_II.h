// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoNSGA_II.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
// (c) Deneche Abdelhakim, 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr
 */
//-----------------------------------------------------------------------------
#ifndef MOEONSGA_II_H_
#define MOEONSGA_II_H_

#include <eoGeneralBreeder.h>
#include <eoBreed.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenContinue.h>
#include <eoGenOp.h>
#include <eoPopEvalFunc.h>
#include <eoSelectOne.h>
#include <eoSGAGenOp.h>

#include <moeoNDSorting.h>
#include <moeoReplacement.h>

/**
*/
template<class EOT> class moeoNSGA_II: public eoAlgo<EOT>  {
public:

	/**
	This constructor builds the algorithm as descibed in the paper

	Deb, K., S. Agrawal, A. Pratap, and T. Meyarivan, 
	A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II. 
	In IEEE Transactions on Evolutionary Computation, Vol. 6, No 2, pp 182-197, April 2002.
	@param _max_gen number of generations before stopping
	@param _eval evaluation function
	@param _op variation operator
	*/

	moeoNSGA_II(
		unsigned _max_gen,
		eoEvalFunc<EOT>& _eval,
		eoGenOp<EOT>& _op
	): continuator(*(new eoGenContinue<EOT>(_max_gen))),
	eval(_eval),
	loopEval(_eval),
	popEval(loopEval),
	selectOne(sorting, 2), // binary tournament selection
	replace(sorting),
	genBreed(selectOne, _op),
	breed(genBreed)
	{}

	/// Ctor taking _max_gen, crossover and mutation
	moeoNSGA_II(
		unsigned _max_gen,
		eoEvalFunc<EOT>& _eval,
		eoQuadOp<EOT>& crossover,
		double pCross,
		eoMonOp<EOT>& mutation,
		double pMut
	): continuator(*(new eoGenContinue<EOT>(_max_gen))),
	eval(_eval),
	loopEval(_eval),
	popEval(loopEval),
	selectOne(sorting, 2), // binary tournament selection
	replace(sorting),
	genBreed(selectOne, *new eoSGAGenOp<EOT>(crossover, pCross, mutation, pMut)),
	breed(genBreed)
	{}

	/// Ctor taking a continuator instead of _gen_max
	moeoNSGA_II(
		eoContinue<EOT>& _continuator,
		eoEvalFunc<EOT>& _eval,
		eoGenOp<EOT>& _op
	): 
	continuator(_continuator),
	eval (_eval),
	loopEval(_eval),
	popEval(loopEval),
	selectOne(sorting, 2), // binary tournament selection
	replace(sorting),
	genBreed(selectOne, _op),
	breed(genBreed)
	{}

	///Apply a few generation of evolution to the population.
  	virtual void operator()(eoPop<EOT>& _pop) 
	{
		eoPop<EOT> offspring, empty_pop;
		popEval(empty_pop, _pop); // a first eval of _pop
		do
		{
			// generate offspring, worths are recalculated if necessary
			breed(_pop, offspring);
			
			// eval of offspring
			popEval(_pop, offspring);

			// after replace, the new pop is in _pop. Worths are recalculated if necessary
			replace(_pop, offspring);
	
		} while (continuator(_pop));
  	}

protected:
  	eoContinue<EOT>&     				continuator;
  	
	eoEvalFunc<EOT>&        			eval;
  	eoPopLoopEval<EOT>        			loopEval;

  	eoPopEvalFunc<EOT>&       			popEval;
	
	/// NSGAII sorting
	moeoNDSorting_II<EOT>				sorting;
	/// Binary tournament selection
	eoDetTournamentWorthSelect<EOT>		selectOne;
	/// Elitist replacement
	moeoElitistReplacement<EOT>			replace;
	eoGeneralBreeder<EOT>				genBreed;
	eoBreed<EOT>&						breed;
};

#endif
