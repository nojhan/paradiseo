// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEasyEA.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _MOEOEASYEA_H
#define _MOEOEASYEA_H

#include <apply.h>
#include <eoPopEvalFunc.h>
#include <eoContinue.h>
#include <eoTransform.h>
#include <eoBreed.h>
#include <eoMergeReduce.h>
#include <moeoEA.h>
#include <eoReplacement.h>
#include <moeoFitnessAssignment.h>
#include <moeoDiversityAssignment.h>

/**
 * An easy class to design multi-objective evolutionary algorithms.
 */
template < class MOEOT > 
class moeoEasyEA: public moeoEA < MOEOT >
{
public:

	/**
	 * Ctor.
	 * @param _continuator the stopping criteria
	 * @param _eval the evaluation functions
	 * @param _breed the breeder
	 * @param _replace the replacment strategy
	 * @param _fitnessEval the fitness evaluation scheme
	 * @param _diversityEval the diversity evaluation scheme
	 * @param _evalFitAndDivBeforeSelection put this parameter to 'true' if you want to re-evalue the fitness and the diversity of the population before the selection process
	 */
	 moeoEasyEA(eoContinue < MOEOT > & _continuator, eoEvalFunc < MOEOT > & _eval, eoBreed < MOEOT > & _breed, eoReplacement < MOEOT > & _replace, 
	 moeoFitnessAssignment < MOEOT > & _fitnessEval, moeoDiversityAssignment < MOEOT > & _diversityEval, bool _evalFitAndDivBeforeSelection = false)
	 : 
	 continuator(_continuator), eval (_eval), loopEval(_eval), popEval(loopEval), breed(_breed), replace(_replace), fitnessEval(_fitnessEval), 
	 diversityEval(_diversityEval), evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
	 {}


	 /**
	  * Applies a few generation of evolution to the population _pop.
	  * @param _pop the population
	  */
	  virtual void operator()(eoPop < MOEOT > & _pop)
	  {
	  	eoPop < MOEOT > offspring, empty_pop;
	  	popEval(empty_pop, _pop); // A first eval of pop.
	  	bool firstTime = true;
	  	do
	  	{
	  		try
	  		{
	  			unsigned pSize = _pop.size();
	  			offspring.clear(); // new offspring
	  			// fitness and diversity assignment (if you want to or if it is the first generation)
	  			if (evalFitAndDivBeforeSelection || firstTime)
	  			{
	  				firstTime = false;
	  				fitnessEval(_pop);
	  				diversityEval(_pop);
	  			}
	  			breed(_pop, offspring);
	  			popEval(_pop, offspring); // eval of parents + offspring if necessary
	  			replace(_pop, offspring); // after replace, the new pop. is in _pop
	  			if (pSize > _pop.size())
	  			{
	  				throw std::runtime_error("Population shrinking!");
	  			}
	  			else if (pSize < _pop.size())
	  			{
	  				throw std::runtime_error("Population growing!");
	  			}
	  		}
	  		catch (std::exception& e)
	  		{
	  			std::string s = e.what();
	    		s.append( " in moeoEasyEA");
	    		throw std::runtime_error( s );
	  		}
	  	} while (continuator(_pop));
	  }


protected:

	/** the stopping criteria */
	eoContinue < MOEOT > & continuator;
	/** the evaluation functions */
	eoEvalFunc < MOEOT > & eval;
	/** to evaluate the whole population */
	eoPopLoopEval < MOEOT > loopEval;
	/** to evaluate the whole population */
	eoPopEvalFunc < MOEOT > & popEval;
	/** the breeder */
	eoBreed < MOEOT > & breed;
	/** the replacment strategy */
	eoReplacement < MOEOT > & replace;
	/** the fitness assignment strategy */
	moeoFitnessAssignment < MOEOT > & fitnessEval;
	/** the diversity assignment strategy */
	moeoDiversityAssignment < MOEOT > & diversityEval;
	/** if this parameter is set to 'true', the fitness and the diversity of the whole population will be re-evaluated before the selection process */
	bool evalFitAndDivBeforeSelection;

};

#endif /*MOEOEASYEA_H_*/
