// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCrowdingDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCROWDINGDISTANCEASSIGNMENT_H_
#define MOEOCROWDINGDISTANCEASSIGNMENT_H_

#include <eoPop.h>
#include <moeoComparator.h>
#include <moeoDiversityAssignment.h>

/**
 * Diversity assignment sheme based on crowding distance proposed in: 
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * This strategy is, for instance, used in NSGA-II.
 */
template < class MOEOT >
class moeoCrowdingDiversityAssignment : public moeoDiversityAssignment < MOEOT >
{
public:


  /** Infinity value */
  double inf ()const
  {
    return std::numeric_limits<double>::max();
  }


	/**
	 * ...
	 * @param _pop the population
	 */
	void operator()(eoPop < MOEOT > & _pop)
	{
		// number of objectives for the problem under consideration
		unsigned nObjectives = MOEOT::ObjectiveVector::nObjectives();
		if (_pop.size() <= 2)
		{
			for (unsigned i=0; i<_pop.size(); i++)
			{
			  _pop[i].diversity(inf());
			}
		}
		else
		{
			setDistances(_pop);
		}
	}


	

private:

	/** the objective vector type of the solutions */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;


	/**
	 * ...
	 * @param _pop the population
	 */
	void setDistances (eoPop < MOEOT > & _pop)
	{
		double min, max, distance;
		unsigned nObjectives = MOEOT::ObjectiveVector::nObjectives();
		// set diversity to 0
		for (unsigned i=0; i<_pop.size(); i++)
		{
			_pop[i].diversity(0);
		}
		// for each objective
		for (unsigned obj=0; obj<nObjectives; obj++)
		{
			// comparator
			moeoOneObjectiveComparator < MOEOT > comp(obj);
			// sort
			std::sort(_pop.begin(), _pop.end(), comp);
			// min & max
			min = _pop[0].objectiveVector()[obj];
			max = _pop[_pop.size()-1].objectiveVector()[obj];
			_pop[0].diversity(inf());
			_pop[_pop.size()-1].diversity(inf());
			for (unsigned i=1; i<_pop.size()-1; i++)
			{
				distance = (_pop[i+1].objectiveVector()[obj] - _pop[i-1].objectiveVector()[obj]) / (max-min);
				_pop[i].diversity(_pop[i].diversity() + distance);
			}
		}
	}

};

#endif /*MOEOCROWDINGDIVERSITYASSIGNMENT_H_*/
