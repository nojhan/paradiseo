/*
* <moeoReferencePointIndicatorBasedFitnessAssignment.h>
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

#ifndef MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_

#include <math.h>
#include <eoPop.h>
#include <fitness/moeoFitnessAssignment.h>
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
    moeoReferencePointIndicatorBasedFitnessAssignment (ObjectiveVector & _refPoint, moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric) :
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
    ObjectiveVector & refPoint;
    /** the quality indicator */
    moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & metric;


    /**
     * Sets the bounds for every objective using the min and the max value for every objective vector of _pop (and the reference point)
     * @param _pop the population
     */
    void setup(const eoPop < MOEOT > & _pop)
    {
      double min, max;
      for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
          min = refPoint[i];
          max = refPoint[i];
          for (unsigned int j=0; j<_pop.size(); j++)
            {
              min = std::min(min, _pop[j].objectiveVector()[i]);
              max = std::max(max, _pop[j].objectiveVector()[i]);
            }
          // setting of the bounds for the objective i
          metric.setup(min, max, i);
        }
    }

    /**
     * Sets the fitness of every individual contained in the population _pop
     * @param _pop the population
     */
    void setFitnesses(eoPop < MOEOT > & _pop)
    {
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].fitness(- metric(_pop[i].objectiveVector(), refPoint) );
        }
    }

  };

#endif /*MOEOREFERENCEPOINTINDICATORBASEDFITNESSASSIGNMENT_H_*/
