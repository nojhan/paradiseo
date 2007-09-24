// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// make_ea_moeo.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MAKE_EA_MOEO_H_
#define MAKE_EA_MOEO_H_

#include <stdlib.h>
#include <eoContinue.h>
#include <eoEvalFunc.h>
#include <eoGeneralBreeder.h>
#include <eoGenOp.h>
#include <utils/eoParser.h>
#include <utils/eoState.h>

#include <algo/moeoEA.h>
#include <algo/moeoEasyEA.h>
#include <archive/moeoArchive.h>
#include <comparator/moeoAggregativeComparator.h>
#include <comparator/moeoComparator.h>
#include <comparator/moeoDiversityThenFitnessComparator.h>
#include <comparator/moeoFitnessThenDiversityComparator.h>
#include <diversity/moeoDiversityAssignment.h>
#include <diversity/moeoDummyDiversityAssignment.h>
#include <diversity/moeoFrontByFrontCrowdingDiversityAssignment.h>
#include <diversity/moeoFrontByFrontSharingDiversityAssignment.h>
#include <fitness/moeoDummyFitnessAssignment.h>
#include <fitness/moeoExpBinaryIndicatorBasedFitnessAssignment.h>
#include <fitness/moeoFastNonDominatedSortingFitnessAssignment.h>
#include <fitness/moeoFitnessAssignment.h>
#include <metric/moeoAdditiveEpsilonBinaryMetric.h>
#include <metric/moeoHypervolumeBinaryMetric.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <replacement/moeoElitistReplacement.h>
#include <replacement/moeoEnvironmentalReplacement.h>
#include <replacement/moeoGenerationalReplacement.h>
#include <replacement/moeoReplacement.h>
#include <selection/moeoDetTournamentSelect.h>
#include <selection/moeoRandomSelect.h>
#include <selection/moeoStochTournamentSelect.h>
#include <selection/moeoSelectOne.h>
#include <selection/moeoSelectors.h>


/**
 * This functions allows to build a moeoEA from the parser
 * @param _parser the parser
 * @param _state to store allocated objects
 * @param _eval the funtions evaluator
 * @param _continue the stopping crietria
 * @param _op the variation operators
 * @param _archive the archive of non-dominated solutions
 */
template < class MOEOT >
moeoEA < MOEOT > & do_make_ea_moeo(eoParser & _parser, eoState & _state, eoEvalFunc < MOEOT > & _eval, eoContinue < MOEOT > & _continue, eoGenOp < MOEOT > & _op, moeoArchive < MOEOT > & _archive)
{

    /* the objective vector type */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /* the fitness assignment strategy */
    std::string & fitnessParam = _parser.createParam(std::string("FastNonDominatedSorting"), "fitness",
                            "Fitness assignment scheme: Dummy, FastNonDominatedSorting or IndicatorBased", 'F',
                            "Evolution Engine").value();
    std::string & indicatorParam = _parser.createParam(std::string("Epsilon"), "indicator",
                              "Binary indicator for IndicatorBased: Epsilon, Hypervolume", 'i',
                              "Evolution Engine").value();
    double rho = _parser.createParam(1.1, "rho", "reference point for the hypervolume indicator", 'r',
                                     "Evolution Engine").value();
    double kappa = _parser.createParam(0.05, "kappa", "Scaling factor kappa for IndicatorBased", 'k',
                                       "Evolution Engine").value();
    moeoFitnessAssignment < MOEOT > * fitnessAssignment;
    if (fitnessParam == std::string("Dummy"))
    {
        fitnessAssignment = new moeoDummyFitnessAssignment < MOEOT> ();
    }
    else if (fitnessParam == std::string("FastNonDominatedSorting"))
    {
        fitnessAssignment = new moeoFastNonDominatedSortingFitnessAssignment < MOEOT> ();
    }
    else if (fitnessParam == std::string("IndicatorBased"))
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
        fitnessAssignment = new moeoExpBinaryIndicatorBasedFitnessAssignment < MOEOT > (*metric, kappa);
    }
    else
    {
        std::string stmp = std::string("Invalid fitness assignment strategy: ") + fitnessParam;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(fitnessAssignment);


    /* the diversity assignment strategy */
    eoValueParam<eoParamParamType> & diversityParam = _parser.createParam(eoParamParamType("Dummy"), "diversity",
            "Diversity assignment scheme: Dummy, Sharing(nicheSize) or Crowding", 'D', "Evolution Engine");
    eoParamParamType & diversityParamValue = diversityParam.value();
    moeoDiversityAssignment < MOEOT > * diversityAssignment;
    if (diversityParamValue.first == std::string("Dummy"))
    {
        diversityAssignment = new moeoDummyDiversityAssignment < MOEOT> ();
    }
    else if (diversityParamValue.first == std::string("Sharing"))
    {
        double nicheSize;
        if (!diversityParamValue.second.size())   // no parameter added
        {
            std::cerr << "WARNING, no niche size given for Sharing, using 0.5" << std::endl;
            nicheSize = 0.5;
            diversityParamValue.second.push_back(std::string("0.5"));
        }
        else
        {
            nicheSize = atoi(diversityParamValue.second[0].c_str());
        }
        diversityAssignment = new moeoFrontByFrontSharingDiversityAssignment < MOEOT> (nicheSize);
    }
    else if (diversityParamValue.first == std::string("Crowding"))
    {
        diversityAssignment = new moeoFrontByFrontCrowdingDiversityAssignment < MOEOT> ();
    }
    else
    {
        std::string stmp = std::string("Invalid diversity assignment strategy: ") + diversityParamValue.first;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(diversityAssignment);


    /* the comparator strategy */
    std::string & comparatorParam = _parser.createParam(std::string("FitnessThenDiversity"), "comparator",
                               "Comparator scheme: FitnessThenDiversity, DiversityThenFitness or Aggregative", 'C', "Evolution Engine").value();
    moeoComparator < MOEOT > * comparator;
    if (comparatorParam == std::string("FitnessThenDiversity"))
    {
        comparator = new moeoFitnessThenDiversityComparator < MOEOT> ();
    }
    else if (comparatorParam == std::string("DiversityThenFitness"))
    {
        comparator = new moeoDiversityThenFitnessComparator < MOEOT> ();
    }
    else if (comparatorParam == std::string("Aggregative"))
    {
        comparator = new moeoAggregativeComparator < MOEOT> ();
    }
    else
    {
        std::string stmp = std::string("Invalid comparator strategy: ") + comparatorParam;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(comparator);


    /* the selection strategy */
    eoValueParam < eoParamParamType > & selectionParam = _parser.createParam(eoParamParamType("DetTour(2)"), "selection",
            "Selection scheme: DetTour(T), StochTour(t) or Random", 'S', "Evolution Engine");
    eoParamParamType & ppSelect = selectionParam.value();
    moeoSelectOne < MOEOT > * select;
    if (ppSelect.first == std::string("DetTour"))
    {
        unsigned int tSize;
        if (!ppSelect.second.size()) // no parameter added
        {
            std::cerr << "WARNING, no parameter passed to DetTour, using 2" << std::endl;
            tSize = 2;
            // put back 2 in parameter for consistency (and status file)
            ppSelect.second.push_back(std::string("2"));
        }
        else // parameter passed by user as DetTour(T)
        {
            tSize = atoi(ppSelect.second[0].c_str());
        }
        select = new moeoDetTournamentSelect < MOEOT > (*comparator, tSize);
    }
    else if (ppSelect.first == std::string("StochTour"))
    {
        double tRate;
        if (!ppSelect.second.size()) // no parameter added
        {
            std::cerr << "WARNING, no parameter passed to StochTour, using 1" << std::endl;
            tRate = 1;
            // put back 1 in parameter for consistency (and status file)
            ppSelect.second.push_back(std::string("1"));
        }
        else // parameter passed by user as StochTour(T)
        {
            tRate = atof(ppSelect.second[0].c_str());
        }
        select = new moeoStochTournamentSelect < MOEOT > (*comparator, tRate);
    }
    /*
    else if (ppSelect.first == string("Roulette"))
    {
        // TO DO !
        // ...
    }
    */
    else if (ppSelect.first == std::string("Random"))
    {
        select = new moeoRandomSelect <MOEOT > ();
    }
    else
    {
        std::string stmp = std::string("Invalid selection strategy: ") + ppSelect.first;
        throw std::runtime_error(stmp.c_str());
    }
    _state.storeFunctor(select);


    /* the replacement strategy */
    std::string & replacementParam = _parser.createParam(std::string("Elitist"), "replacement",
                                "Replacement scheme: Elitist, Environmental or Generational", 'R', "Evolution Engine").value();
    moeoReplacement < MOEOT > * replace;
    if (replacementParam == std::string("Elitist"))
    {
        replace = new moeoElitistReplacement < MOEOT> (*fitnessAssignment, *diversityAssignment, *comparator);
    }
    else if (replacementParam == std::string("Environmental"))
    {
        replace = new moeoEnvironmentalReplacement < MOEOT> (*fitnessAssignment, *diversityAssignment, *comparator);
    }
    else if (replacementParam == std::string("Generational"))
    {
        replace = new moeoGenerationalReplacement < MOEOT> ();
    }
    else
    {
        std::string stmp = std::string("Invalid replacement strategy: ") + replacementParam;
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
    moeoEA < MOEOT > * algo = new moeoEasyEA < MOEOT > (_continue, _eval, *breed, *replace, *fitnessAssignment, *diversityAssignment);
    _state.storeFunctor(algo);
    return *algo;

}

#endif /*MAKE_EA_MOEO_H_*/
