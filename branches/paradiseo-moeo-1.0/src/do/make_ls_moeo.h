// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_ls_moeo.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_LS_MOEO_H_
#define MAKE_LS_MOEO_H_

#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGenOp.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>
#include <moeoArchive.h>
#include <moeoIndicatorBasedFitnessAssignment.h>
#include <moeoLS.h>
#include <moeoIndicatorBasedLS.h>
#include <moeoIteratedIBMOLS.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <moeoMoveIncrEval.h>

/**
 * ...
 */
template < class MOEOT, class Move >
moeoLS < MOEOT, eoPop<MOEOT> & > & do_make_ls_moeo	(
					eoParser & _parser,
					eoState & _state,
					moeoEvalFunc < MOEOT > & _eval,
					moeoMoveIncrEval < Move > & _moveIncrEval,
					eoContinue < MOEOT > & _continue,
					eoMonOp < MOEOT > & _op,
					eoMonOp < MOEOT > & _opInit,
					moMoveInit < Move > & _moveInit,
					moNextMove < Move > & _nextMove,
					moeoArchive < MOEOT > & _archive
					)
{
	
	/* the objective vector type */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;


	/* the fitness assignment strategy */
	string & fitnessParam = _parser.getORcreateParam(string("IndicatorBased"), "fitness", 
		"Fitness assignment strategy parameter: IndicatorBased...", 'F',
		"Evolution Engine").value();
	string & indicatorParam = _parser.getORcreateParam(string("Epsilon"), "indicator", 
		"Binary indicator to use with the IndicatorBased assignment: Epsilon, Hypervolume", 'i',
		"Evolution Engine").value();
	double rho = _parser.getORcreateParam(1.1, "rho", "reference point for the hypervolume indicator", 
		'r', "Evolution Engine").value();
	double kappa = _parser.getORcreateParam(0.05, "kappa", "Scaling factor kappa for IndicatorBased",
		'k', "Evolution Engine").value();
	moeoIndicatorBasedFitnessAssignment < MOEOT > * fitnessAssignment;
	if (fitnessParam == string("IndicatorBased"))
	{
		// metric
		moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > *metric;
		if (indicatorParam == string("Epsilon"))
		{
			metric = new moeoAdditiveEpsilonBinaryMetric < ObjectiveVector >;
		}
		else if (indicatorParam == string("Hypervolume"))
		{
			metric = new moeoHypervolumeBinaryMetric < ObjectiveVector > (rho);
		}
		else
		{
			string stmp = string("Invalid binary quality indicator: ") + indicatorParam;
      			throw std::runtime_error(stmp.c_str());
		}
		fitnessAssignment = new moeoIndicatorBasedFitnessAssignment < MOEOT> (metric, kappa);
	}
	else
	{
		string stmp = string("Invalid fitness assignment strategy: ") + fitnessParam;
		throw std::runtime_error(stmp.c_str());
	}
	_state.storeFunctor(fitnessAssignment);




	unsigned n = _parser.getORcreateParam(1, "n", "Number of iterations for population Initialization",
		'n', "Evolution Engine").value();



	// LS
	string & lsParam = _parser.getORcreateParam(string("I-IBMOLS"), "ls",
		"Local Search: IBMOLS, I-IBMOLS (Iterated-IBMOLS)...", 'L',
		"Evolution Engine").value();
	moeoLS < MOEOT, eoPop<MOEOT> & > * ls;
	if (lsParam == string("IBMOLS"))
	{
	  ls = new moeoIndicatorBasedLS < MOEOT, Move > (_moveInit, _nextMove, _eval, _moveIncrEval, *fitnessAssignment, _continue);;
	}
	else if (lsParam == string("I-IBMOLS"))
	{
	  ls = new moeoIteratedIBMOLS < MOEOT, Move > (_moveInit, _nextMove, _eval, _moveIncrEval, *fitnessAssignment, _continue, _op, _opInit, n);
	}
	else
	{
		string stmp = string("Invalid fitness assignment strategy: ") + fitnessParam;
		throw std::runtime_error(stmp.c_str());
	}
	_state.storeFunctor(ls);


	return *ls;
	
}

#endif /*MAKE_LS_MOEO_H_*/
