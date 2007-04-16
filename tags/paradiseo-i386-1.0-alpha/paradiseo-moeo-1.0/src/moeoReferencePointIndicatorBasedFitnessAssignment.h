// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoReferencePointIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_

#include <math.h>
#include <eoPop.h>
#include <moeoFitnessAssignment.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>

/**
 * Fitness assignment sheme based a Reference Point and a Quality Indicator.
 */
template < class MOEOT >
class moeoReferencePointIndicatorBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{
public:

    /** The type of objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    /**
     * Ctor
     * @param _refPoint the reference point
     * @param _metric the quality indicator
     */
    moeoReferencePointIndicatorBasedFitnessAssignment (const ObjectiveVector _refPoint, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric) :
            refPoint(_refPoint), metric(_metric)
    {}


    /**
     * Sets the fitness values for every solution contained in the population _pop
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
        // 1 - setting of the bounds
        setup(_pop);
        // 2 - setting fitnesses
        setFitnesses(_pop);
    }


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        // nothing to do  ;-)
    }


protected:

    /** the reference point */
    ObjectiveVector refPoint;
    /** the quality indicator */
    moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * metric;


    /**
     * Sets the bounds for every objective using the min and the max value for every objective vector of _pop (and the reference point)
     * @param _pop the population
     */
    void setup(const eoPop < MOEOT > & _pop)
    {
        double min, max;
        for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            min = refPoint[i];
            max = refPoint[i];
            for (unsigned j=0; j<_pop.size(); j++)
            {
                min = std::min(min, _pop[j].objectiveVector()[i]);
                max = std::max(max, _pop[j].objectiveVector()[i]);
            }
            // setting of the bounds for the objective i
            (*metric).setup(min, max, i);
        }
    }

    /**
     * Sets the fitness of every individual contained in the population _pop
     * @param _pop the population
     */
    void setFitnesses(eoPop < MOEOT > & _pop)
    {
        for (unsigned i=0; i<_pop.size(); i++)
        {
            _pop[i].fitness(- (*metric)(_pop[i].objectiveVector(), refPoint) );
        }
    }

};

#endif /*MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_*/
