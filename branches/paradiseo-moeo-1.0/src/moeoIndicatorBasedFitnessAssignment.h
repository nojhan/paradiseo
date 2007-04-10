// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOINDICATORBASEDFITNESSASSIGNMENT_H_

#include <math.h>
#include <eoPop.h>
#include <moeoConvertPopToObjectiveVectors.h>
#include <moeoFitnessAssignment.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>

/**
 * Default is exponential
 */
template < class MOEOT >
class moeoIndicatorBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{
public:

	/** The type of objective vector */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	/**
	 * Ctor
	 * @param ...
	 */
	moeoIndicatorBasedFitnessAssignment(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric, const double _kappa) : metric(_metric), kappa(_kappa)
	{}
	
	
	/**
	 *
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


/////////////////////////////////////////////////////////////////////


// A SIMPLIFIER ! => utiliser la fonction d'en dessous ;-)
	void updateByDeleting(eoPop < MOEOT > & _pop, MOEOT & _moeo)
	{
		vector < double > v;
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_moeo.objectiveVector(), _pop[i].objectiveVector());
		}
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].fitness( _pop[i].fitness() + exp(-v[i]/kappa) );
		}
	}


        void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
	{
		vector < double > v;
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_objVec, _pop[i].objectiveVector());
		}
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].fitness( _pop[i].fitness() + exp(-v[i]/kappa) );
		}
	}


  // IDEM !
	void updateByAdding(eoPop < MOEOT > & _pop, MOEOT & _moeo)
	{
		vector < double > v;
		// update every fitness values to take the new individual into account
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_moeo.objectiveVector(), _pop[i].objectiveVector());
		}
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].fitness( _pop[i].fitness() - exp(-v[i]/kappa) );
		}
		// compute the fitness of the new individual
		v.clear();
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_pop[i].objectiveVector(), _moeo.objectiveVector());
		}
		double fitness = 0;
		for (unsigned i=0; i<v.size(); i++)
		{
			fitness -= exp(-v[i]/kappa);
		}
		_moeo.fitness(fitness);
	}


  // update _pop et retourne la valeur de fitness de _objVec
        double updateByAdding(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
	{
		vector < double > v;
		// update every fitness values to take the new individual into account
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_objVec, _pop[i].objectiveVector());
		}
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].fitness( _pop[i].fitness() - exp(-v[i]/kappa) );
		}
		// compute the fitness of the new individual
		v.clear();
		v.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			v[i] = (*metric)(_pop[i].objectiveVector(), _objVec);
		}
		double result = 0;
		for (unsigned i=0; i<v.size(); i++)
		{
			result -= exp(-v[i]/kappa);
		}
		return result;
	}


/////////////////////////////////////////////////////////////////////


protected:
	moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * metric;
	double kappa;
	std::vector < std::vector<double> > values;


	/**
	 * Sets the bounds for every objective using the min and the max value for every objective vector of _pop
	 * @param _pop the population
	 */
	void setup(const eoPop < MOEOT > & _pop)
	{
		double min, max;
		for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
		{
			min = _pop[0].objectiveVector()[i];
			max = _pop[0].objectiveVector()[i];
			for (unsigned j=1; j<_pop.size(); j++)
			{
				min = std::min(min, _pop[j].objectiveVector()[i]);
				max = std::max(max, _pop[j].objectiveVector()[i]);
			}
			// setting of the bounds for the objective i
			(*metric).setup(min, max, i);
		}
	}

	/**
	 * Compute every indicator value : values[i] = I(_v[i], _o) !!!!!!!!!!!
	 * @param ...
	 */
	void computeValues(const eoPop < MOEOT > & _pop)
	{
		values.clear();
		values.resize(_pop.size());
		for (unsigned i=0; i<_pop.size(); i++)
		{
			values[i].resize(_pop.size());
			for (unsigned j=0; j<_pop.size(); j++)
			{
				if (i != j)
				{
					values[i][j] = (*metric)(_pop[i].objectiveVector(), _pop[j].objectiveVector());
				}
			}
		}
	}

	void setFitnesses(eoPop < MOEOT > & _pop)
	{
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].fitness(computeFitness(i));
		}
	}

	double computeFitness(const unsigned _idx)
	{
		double result = 0;
		for (unsigned i=0; i<values.size(); i++)
		{
			if (i != _idx)
			{
				result -= exp(-values[i][_idx]/kappa);
			}
		}
		return result;
	}
};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
