// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCrowdingDistanceDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCROWDINGDISTANCEDIVERSITYASSIGNMENT_H_
#define MOEOCROWDINGDISTANCEDIVERSITYASSIGNMENT_H_

#include <eoPop.h>
#include <moeoComparator.h>
#include <moeoDiversityAssignment.h>

/**
 * Diversity assignment sheme based on crowding distance proposed in: 
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * This strategy is, for instance, used in NSGA-II.
 */
template < class MOEOT >
class moeoCrowdingDistanceDiversityAssignment : public moeoDiversityAssignment < MOEOT >
{
public:

	/** the objective vector type of the solutions */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	
	/**
	 * Returns a big value (regarded as infinite)
	 */
	 double inf() const
	 {
	 	return std::numeric_limits<double>::max();
	 }
	 
	 
	 /**
	  * Computes diversity values for every solution contained in the population _pop
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


	/**
	 * @warning NOT IMPLEMENTED, DO NOTHING !
	 * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
	 * @param _pop the population
	 * @param _objVec the objective vector
	 * @warning NOT IMPLEMENTED, DO NOTHING !
	 */
	void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
	{
		cout << "WARNING : updateByDeleting not implemented in moeoCrowdingDiversityAssignment" << endl;
	}
	
	
private:

	/**
	 * Sets the distance values
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
			// set the diversity value to infiny for min and max
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

#endif /*MOEOCROWDINGDISTANCEDIVERSITYASSIGNMENT_H_*/
