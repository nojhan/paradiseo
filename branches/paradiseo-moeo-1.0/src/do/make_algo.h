// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_algo.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_ALGO_H_
#define MAKE_ALGO_H_

#include <eoEasyEA.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGeneralBreeder.h>
#include <eoGenOp.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <moeoArchive.h>
#include <moeoComparator.h>
#include <moeoDetTournamentSelect.h>
#include <moeoDiversityAssignment.h>
#include <moeoEA.h>
#include <moeoElitistReplacement.h>
#include <moeoFastNonDominatedSortingFitnessAssignment.h>
#include <moeoFitnessAssignment.h>
#include <moeoGenerationalReplacement.h>
#include <moeoIndicatorBasedFitnessAssignment.h>
#include <moeoRandomSelect.h>
#include <moeoReplacement.h>
#include <moeoRouletteSelect.h>
#include <moeoSelectOne.h>
#include <moeoStochTournamentSelect.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <metric/moeoVectorVsSolutionBinaryMetric.h>

/**
 * ...
 *  
 * !!! eoEvalFunc => moeoEvalFunc
 * !!! eoAlgo => moeoEA
 * !!! ...
 */
template < class MOEOT >
eoAlgo < MOEOT > & do_make_algo(eoParser & _parser, eoState & _state, eoEvalFunc < MOEOT > & _eval, 
									eoContinue < MOEOT > & _continue, eoGenOp < MOEOT > & _op, moeoArchive < MOEOT > & _archive)
{
	
	/* the fitness assignment strategy */
	string & fitnessParam = _parser.createParam(string("FastNonDominatedSorting"), "fitness", 
		"Fitness assignment strategy parameter: FastNonDominatedSorting, IndicatorBased...", 'F', "Evolution Engine").value();
	moeoFitnessAssignment < MOEOT > * fitnessAssignment;
	if (fitnessParam == string("FastNonDominatedSorting"))
	{
		fitnessAssignment = new moeoFastNonDominatedSortingFitnessAssignment < MOEOT> ();
	}
	/****************************************************************************************************************************/
	else if (fitnessParam == string("IndicatorBased"))
	{
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    	moeoAdditiveEpsilonBinaryMetric < ObjectiveVector > * e = new moeoAdditiveEpsilonBinaryMetric < ObjectiveVector >;
		moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > * metric = new moeoExponentialVectorVsSolutionBinaryMetric < ObjectiveVector> (e,0.001);
		fitnessAssignment = new moeoIndicatorBasedFitnessAssignment < MOEOT> (metric);
	}
	/****************************************************************************************************************************/
    else
    {
    	string stmp = string("Invalid fitness assignment strategy: ") + fitnessParam;
    	throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(fitnessAssignment);
	
	
	/* the diversity assignment strategy */
	string & diversityParam = _parser.createParam(string("Dummy"), "diversity", 
		"Diversity assignment strategy parameter: Dummy, ...", 'D', "Evolution Engine").value();
	moeoDiversityAssignment < MOEOT > * diversityAssignment;
	if (diversityParam == string("Dummy"))
	{
		diversityAssignment = new moeoDummyDiversityAssignment < MOEOT> ();
	}
    else
    {
    	string stmp = string("Invalid diversity assignment strategy: ") + diversityParam;
    	throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(diversityAssignment);
    
    
    /* the comparator strategy */
    string & comparatorParam = _parser.createParam(string("FitnessThenDiversity"), "comparator", 
		"Comparator strategy parameter: FitnessThenDiversity or DiversityThenFitness", 'C', "Evolution Engine").value();
	moeoComparator < MOEOT > * comparator;
	if (comparatorParam == string("FitnessThenDiversity"))
	{
		comparator = new moeoFitnessThenDiversityComparator < MOEOT> ();
	}
	else if (comparatorParam == string("DiversityThenFitness"))
	{
		comparator = new moeoDiversityThenFitnessComparator < MOEOT> ();
	}
    else
    {
    	string stmp = string("Invalid comparator strategy: ") + comparatorParam;
    	throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(comparator);
    
    
    /* the selection strategy */
	eoValueParam < eoParamParamType > & selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection", 
		"Selection strategy parameter: DetTour(T), StochTour(t), Roulette or Random", 'S', "Evolution Engine");
	eoParamParamType & ppSelect = selectionParam.value();
	moeoSelectOne < MOEOT > * select;
	if (ppSelect.first == string("DetTour"))
	{
		unsigned tSize;
		if (!ppSelect.second.size()) // no parameter added
		{
			cerr << "WARNING, no parameter passed to DetTour, using 2" << endl;
      		tSize = 2;
      		// put back 2 in parameter for consistency (and status file)
      		ppSelect.second.push_back(string("2"));
		}
		else // parameter passed by user as DetTour(T)
		{
			tSize = atoi(ppSelect.second[0].c_str());
		}
		select = new moeoDetTournamentSelect < MOEOT > (*fitnessAssignment, *diversityAssignment, *comparator, tSize);
	}
	else if (ppSelect.first == string("StochTour"))
	{
		double tRate;
		if (!ppSelect.second.size()) // no parameter added
		{
			cerr << "WARNING, no parameter passed to StochTour, using 1" << endl;
      		tRate = 1;
      		// put back 1 in parameter for consistency (and status file)
      		ppSelect.second.push_back(string("1"));
		}
		else // parameter passed by user as StochTour(T)
		{
			tRate = atof(ppSelect.second[0].c_str());
		}
		select = new moeoStochTournamentSelect < MOEOT > (*fitnessAssignment, *diversityAssignment, *comparator, tRate);
	}
	else if (ppSelect.first == string("Roulette"))
	{
		// TO DO !
		// ...
	}
	else if (ppSelect.first == string("Random"))
	{
		select = new moeoRandomSelect <MOEOT > ();
	}
	else
	{
		string stmp = string("Invalid selection strategy: ") + ppSelect.first;
		throw std::runtime_error(stmp.c_str());
	}
    _state.storeFunctor(select);
    
    
    /* the replacement strategy */
    string & replacementParam = _parser.createParam(string("Elitist"), "replacement", 
		"Replacement strategy parameter: Elitist or Generational", 'R', "Evolution Engine").value();
	moeoReplacement < MOEOT > * replace;
	if (replacementParam == string("Elitist"))
	{
		replace = new moeoElitistReplacement < MOEOT> (*fitnessAssignment, *diversityAssignment, *comparator);
	}
	else if (replacementParam == string("Generational"))
	{
		replace = new moeoGenerationalReplacement < MOEOT> ();
	}
    else
    {
    	string stmp = string("Invalid replacement strategy: ") + replacementParam;
    	throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(replace);
    
    
	/* the number of offspring  */
	eoValueParam < eoHowMany > & offspringRateParam = _parser.createParam(eoHowMany(1.0), "nbOffspring", 
		"Number of offspring (percentage or absolute)", 'O', "Evolution Engine");
	
	
	// the general breeder
	eoGeneralBreeder < MOEOT > * breed = new eoGeneralBreeder < MOEOT > (*select, _op, offspringRateParam.value());
	_state.storeFunctor(breed);
	// the eoEasyEA
	eoAlgo < MOEOT > * algo = new eoEasyEA < MOEOT > (_continue, _eval, *breed, *replace);
	_state.storeFunctor(algo);
	return *algo;
	
}

#endif /*MAKE_ALGO_H_*/
