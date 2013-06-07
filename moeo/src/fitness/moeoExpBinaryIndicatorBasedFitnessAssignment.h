/*
* <moeoExpBinaryIndicatorBasedFitnessAssignment.h>
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

#ifndef MOEOEXPBINARYINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOEXPBINARYINDICATORBASEDFITNESSASSIGNMENT_H_

#include <math.h>
#include <vector>
#include <eoPop.h>
#include <fitness/moeoBinaryIndicatorBasedFitnessAssignment.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>
#include <utils/moeoConvertPopToObjectiveVectors.h>

/**
 * Fitness assignment sheme based on an indicator proposed in:
 * E. Zitzler, S. Künzli, "Indicator-Based Selection in Multiobjective Search", Proc. 8th International Conference on Parallel Problem Solving from Nature (PPSN VIII), pp. 832-842, Birmingham, UK (2004).
 * This strategy is, for instance, used in IBEA.
 */
template < class MOEOT >
class moeoExpBinaryIndicatorBasedFitnessAssignment : public moeoBinaryIndicatorBasedFitnessAssignment < MOEOT >
  {
  public:

    /** The type of objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor.
     * @param _metric the quality indicator
     * @param _kappa the scaling factor
     */
    moeoExpBinaryIndicatorBasedFitnessAssignment(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & _metric, const double _kappa = 0.05) : metric(_metric), kappa(_kappa)
    {}


    /**
     * Sets the fitness values for every solution contained in the population _pop
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
      // 1 - setting of the bounds
      setup(_pop);
      // 2 - computing every indicator values
      computeValues(_pop);
      // 3 - setting fitnesses
      setFitnesses(_pop);
    }


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
      std::vector < double > v;
      v.resize(_pop.size());
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          v[i] = metric(_objVec, _pop[i].objectiveVector());
        }
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].fitness( _pop[i].fitness() + exp(-v[i]/kappa) );
        }
    }


    /**
     * Updates the fitness values of the whole population _pop by taking the adding of the objective vector _objVec into account
     * and returns the fitness value of _objVec.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    double updateByAdding(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
      std::vector < double > v;
      // update every fitness values to take the new individual into account
      v.resize(_pop.size());
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          v[i] = metric(_objVec, _pop[i].objectiveVector());
        }
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].fitness( _pop[i].fitness() - exp(-v[i]/kappa) );
        }
      // compute the fitness of the new individual
      v.clear();
      v.resize(_pop.size());
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          v[i] = metric(_pop[i].objectiveVector(), _objVec);
        }
      double result = 0;
      for (unsigned int i=0; i<v.size(); i++)
        {
          result -= exp(-v[i]/kappa);
        }
      return result;
    }


  protected:

    /** the quality indicator */
    moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > & metric;
    /** the scaling factor */
    double kappa;
    /** the computed indicator values */
    std::vector < std::vector<double> > values;


    /**
     * Sets the bounds for every objective using the min and the max value for every objective vector of _pop
     * @param _pop the population
     */
    void setup(const eoPop < MOEOT > & _pop)
    {
      typename MOEOT::ObjectiveVector::Type min, max;
      for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
          min = _pop[0].objectiveVector()[i];
          max = _pop[0].objectiveVector()[i];
          for (unsigned int j=1; j<_pop.size(); j++)
            {
              min = std::min(min, _pop[j].objectiveVector()[i]);
              max = std::max(max, _pop[j].objectiveVector()[i]);
            }
          // setting of the bounds for the objective i
          metric.setup(min, max, i);
        }
    }


    /**
     * Compute every indicator value in values (values[i] = I(_v[i], _o))
     * @param _pop the population
     */
    void computeValues(const eoPop < MOEOT > & _pop)
    {
      values.clear();
      values.resize(_pop.size());
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          values[i].resize(_pop.size());
          for (unsigned int j=0; j<_pop.size(); j++)
            {
              if (i != j)
                {
                  values[i][j] = metric(_pop[i].objectiveVector(), _pop[j].objectiveVector());
                }
            }
        }
    }


    /**
     * Sets the fitness value of the whple population
     * @param _pop the population
     */
    void setFitnesses(eoPop < MOEOT > & _pop)
    {
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].fitness(computeFitness(i));
        }
    }


    /**
     * Returns the fitness value of the _idx th individual of the population
     * @param _idx the index
     */
    double computeFitness(const unsigned int _idx)
    {
      double result = 0;
      for (unsigned int i=0; i<values.size(); i++)
        {
          if (i != _idx)
            {
              result -= exp(-values[i][_idx]/kappa);
            }
        }
      return result;
    }

  };

#endif /*MOEOEXPBINARYINDICATORBASEDFITNESSASSIGNMENT_H_*/
