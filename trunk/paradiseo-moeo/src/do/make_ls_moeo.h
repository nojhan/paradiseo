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
#include <algo/moeoIBMOLS.h>
#include <algo/moeoIteratedIBMOLS.h>
#include <algo/moeoLS.h>
#include <archive/moeoArchive.h>
#include <fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <fitness/moeoExpBinaryIndicatorBasedFitnessAssignment.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <move/moeoMoveIncrEval.h>

/**
 * This functions allows to build a moeoLS from the parser
 * @param _parser the parser
 * @param _state to store allocated objects
 * @param _eval the funtions evaluator
 * @param _moveIncrEval the incremental evaluation
 * @param _continue the stopping crietria
 * @param _op the variation operators
 * @param _opInit the initilization operator
 * @param _moveInit the move initializer
 * @param _nextMove the move incrementor
 * @param _archive the archive of non-dominated solutions
 */
template < class MOEOT, class Move >
moeoLS < MOEOT, eoPop<MOEOT> & > & do_make_ls_moeo	(
    eoParser & _parser,
    eoState & _state,
    eoEvalFunc < MOEOT > & _eval,
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
    std::string & fitnessParam = _parser.getORcreateParam(std::string("IndicatorBased"), "fitness",
                            "Fitness assignment strategy parameter: IndicatorBased...", 'F',
                            "Evolution Engine").value();
    std::string & indicatorParam = _parser.getORcreateParam(std::string("Epsilon"), "indicator",
                              "Binary indicator to use with the IndicatorBased assignment: Epsilon, Hypervolume", 'i',
                              "Evolution Engine").value();
    double rho = _parser.getORcreateParam(1.1, "rho", "reference point for the hypervolume indicator",
                                          'r', "Evolution Engine").value();
    double kappa = _parser.getORcreateParam(0.05, "kappa", "Scaling factor kappa for IndicatorBased",
                                            'k', "Evolution Engine").value();
    moeoBinaryIndicatorBasedFitnessAssignment < MOEOT > * fitnessAssignment;
    if (fitnessParam == std::string("IndicatorBased"))
    {
        // metric
        moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > *metric;
        if (indicatorParam == std::string("Epsilon"))
        {
            metric = new moeoAdditiveEpsilonBinaryMetric < ObjectiveVector >;
        }
        else if (indicatorParam == std::string("Hypervolume"))
        {
            metric = new moeoHypervolumeBinaryMetric < ObjectiveVector > (rho);
        }
        else
        {
            std::string stmp = std::string("Invalid binary quality indicator: ") + indicatorParam;
            throw std::runtime_error(stmp.c_str());
        }
        fitnessAssignment = new moeoExpBinaryIndicatorBasedFitnessAssignment < MOEOT> (*metric, kappa);
    }
    else
    {
        std::string stmp = std::string("Invalid fitness assignment strategy: ") + fitnessParam;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(fitnessAssignment);
    // number of iterations
    unsigned int n = _parser.getORcreateParam(1, "n", "Number of iterations for population Initialization", 'n', "Evolution Engine").value();
    // LS
    std::string & lsParam = _parser.getORcreateParam(std::string("I-IBMOLS"), "ls",
                       "Local Search: IBMOLS, I-IBMOLS (Iterated-IBMOLS)...", 'L',
                       "Evolution Engine").value();
    moeoLS < MOEOT, eoPop<MOEOT> & > * ls;
    if (lsParam == std::string("IBMOLS"))
    {
        ls = new moeoIBMOLS < MOEOT, Move > (_moveInit, _nextMove, _eval, _moveIncrEval, *fitnessAssignment, _continue);;
    }
    else if (lsParam == std::string("I-IBMOLS"))
    {
        ls = new moeoIteratedIBMOLS < MOEOT, Move > (_moveInit, _nextMove, _eval, _moveIncrEval, *fitnessAssignment, _continue, _op, _opInit, n);
    }
    else
    {
        std::string stmp = std::string("Invalid fitness assignment strategy: ") + fitnessParam;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(ls);
    // that's it !
    return *ls;
}

#endif /*MAKE_LS_MOEO_H_*/
