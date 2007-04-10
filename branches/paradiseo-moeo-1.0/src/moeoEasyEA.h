// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

#ifndef _MOEOEASYEA_H
#define _MOEOEASYEA_H

//-----------------------------------------------------------------------------

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

template < class MOEOT > 
class moeoEasyEA: public moeoEA < MOEOT >
{
public:

  /** Ctor taking a breed and merge */
     moeoEasyEA(
         eoContinue<MOEOT>& _continuator,
         eoEvalFunc<MOEOT>& _eval,
         eoBreed<MOEOT>& _breed,
         eoReplacement<MOEOT>& _replace,
         moeoFitnessAssignment<MOEOT>& _fitnessEval,
         moeoDiversityAssignment<MOEOT>& _diversityEval,
         bool _evalFitAndDivBeforeSelection = false
     ) : continuator(_continuator),
	 eval (_eval),
	 loopEval(_eval),
	 popEval(loopEval),
         breed(_breed),
         replace(_replace),
	fitnessEval(_fitnessEval),
	diversityEval(_diversityEval),
	evalFitAndDivBeforeSelection(_evalFitAndDivBeforeSelection)
         {}

  /// Apply a few generation of evolution to the population.
  virtual void operator()(eoPop<MOEOT>& _pop)
  {
    eoPop<MOEOT> offspring, empty_pop;
    popEval(empty_pop, _pop); // A first eval of pop.
    fitnessEval(_pop);
    diversityEval(_pop);
    bool firstTime = true;
    do
    {
      try
      {
         unsigned pSize = _pop.size();
         offspring.clear(); // new offspring

	/************************************/
	if ( evalFitAndDivBeforeSelection && (! firstTime) )
	{
		fitnessEval(_pop);
		diversityEval(_pop);
	}
	else
	{
		firstTime = false;
	}
	/************************************/
	
         breed(_pop, offspring);

         popEval(_pop, offspring); // eval of parents + offspring if necessary

         replace(_pop, offspring); // after replace, the new pop. is in _pop

         if (pSize > _pop.size())
             throw std::runtime_error("Population shrinking!");
         else if (pSize < _pop.size())
             throw std::runtime_error("Population growing!");
      }
      catch (std::exception& e)
      {
	    std::string s = e.what();
	    s.append( " in moeoEasyEA");
	    throw std::runtime_error( s );
      }
    } while ( continuator( _pop ) );

  }


protected :

  eoContinue<MOEOT>&          continuator;
  eoEvalFunc <MOEOT> &        eval ;
  eoPopLoopEval<MOEOT>        loopEval;
  eoPopEvalFunc<MOEOT>&       popEval;
  eoBreed<MOEOT>&             breed;
  eoReplacement<MOEOT>&       replace;
  /** the fitness assignment strategy */
  moeoFitnessAssignment < MOEOT > & fitnessEval;
  /** the diversity assignment strategy */
  moeoDiversityAssignment < MOEOT > & diversityEval;
  /** */
  bool evalFitAndDivBeforeSelection;

};

//-----------------------------------------------------------------------------

#endif
