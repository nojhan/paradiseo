// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFastNonDominatedSortingFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFASTNONDOMINATEDSORTINGFITNESSASSIGNMENT_H_
#define MOEOFASTNONDOMINATEDSORTINGFITNESSASSIGNMENT_H_

#include <eoPop.h>
#include <moeoFitnessAssignment.h>
#include <moeoComparator.h>
#include <moeoObjectiveVectorComparator.h>

/**
 * Fitness assignment sheme based on Pareto-dominance count proposed in 
 * N. Srinivas, K. Deb, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms", Evolutionary Computation vol. 2, no. 3, pp. 221-248 (1994)
 * and in
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * This strategy is, for instance, used in NSGA and NSGA-II.
 */
template < class MOEOT > class moeoFastNonDominatedSortingFitnessAssignment:public moeoParetoBasedFitnessAssignment <
  MOEOT
  >
{
public:

	/**
	 * Ctor
	 */
  moeoFastNonDominatedSortingFitnessAssignment ()
  {
  }


	/**
	 * Computes fitness values for every solution contained in the population _pop
	 * @param _pop the population
	 */
  void operator () (eoPop < MOEOT > &_pop)
  {
    // number of objectives for the problem under consideration
    unsigned nObjectives = MOEOT::ObjectiveVector::nObjectives ();
    if (nObjectives == 1)
      {
	// one objective
	oneObjective (_pop);
      }
    else if (nObjectives == 2)
      {
	// two objectives (the two objectives function is still to do)
	mObjectives (_pop);
      }
    else if (nObjectives > 2)
      {
	// more than two objectives
	mObjectives (_pop);
      }
    else
      {
	// problem with the number of objectives
	throw std::
	  runtime_error
	  ("Problem with the number of objectives in moeoFastNonDominatedSortingFitnessAssignment");
      }
  }


private:

	/** the objective vector type of the solutions */
  typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	/** Functor to compare two objective vectors according to Pareto dominance relation */
  moeoParetoObjectiveVectorComparator < ObjectiveVector > comparator;
	/** Functor to compare two solutions on the first objective, then on the second, and so on */
  moeoObjectiveComparator < MOEOT > objComparator;


	/**
	 * Sets the fitness values for mono-objective problems
	 * @param _pop the population
	 */
  void oneObjective (eoPop < MOEOT > &_pop)
  {
    std::sort (_pop.begin (), _pop.end (), objComparator);
    for (unsigned i = 0; i < _pop.size (); i++)
      {
	_pop[i].fitness (i + 1);
      }
  }


	/**
	 * Sets the fitness values for bi-objective problems with a complexity of O(n log n), where n stands for the population size
	 * @param _pop the population
	 */
  void twoObjectives (eoPop < MOEOT > &_pop)
  {
    //... TO DO !
  }


	/**
	 * Sets the fitness values for problems with more than two objectives with a complexity of O(n² log n), where n stands for the population size
	 * @param _pop the population
	 */
  void mObjectives (eoPop < MOEOT > &_pop)
  {
    // S[i] = indexes of the individuals dominated by _pop[i]
    std::vector < std::vector < unsigned >>S (_pop.size ());
    // n[i] = number of individuals that dominate the individual _pop[i]
    std::vector < unsigned >n (_pop.size (), 0);
    // fronts: F[i] = indexes of the individuals contained in the ith front
    std::vector < std::vector < unsigned >>F (_pop.size () + 1);
    // used to store the number of the first front
    F[1].reserve (_pop.size ());
    // flag to comparae solutions
    int comparatorFlag;
    for (unsigned p = 0; p < _pop.size (); p++)
      {
	for (unsigned q = 0; q < _pop.size (); q++)
	  {
	    // comparison of the 2 solutions according to Pareto dominance
	    comparatorFlag =
	      comparator (_pop[p].objectiveVector (),
			  _pop[q].objectiveVector ());
	    // if p dominates q
	    if (comparatorFlag == 1)
	      {
		// add q to the set of solutions dominated by p
		S[p].push_back (q);
	      }
	    // if q dominates p
	    else if (comparatorFlag == -1)
	      {
		// increment the domination counter of p
		n[p]++;
	      }
	  }
	// if no individual dominates p
	if (n[p] == 0)
	  {
	    // p belongs to the first front
	    _pop[p].fitness (1);
	    F[1].push_back (p);
	  }
      }
    // front counter
    unsigned counter = 1;
    unsigned p, q;
    while (!F[counter].empty ())
      {
	// used to store the number of the next front
	F[counter + 1].reserve (_pop.size ());
	for (unsigned i = 0; i < F[counter].size (); i++)
	  {
	    p = F[counter][i];
	    for (unsigned j = 0; j < S[p].size (); j++)
	      {
		q = S[p][j];
		n[q]--;
		// if no individual dominates q anymore
		if (n[q] == 0)
		  {
		    // q belongs to the next front
		    _pop[q].fitness (counter + 1);
		    F[counter + 1].push_back (q);
		  }
	      }
	  }
	counter++;
      }
  }

};

#endif /*MOEOFASTNONDOMINATEDSORTINGFITNESSASSIGNMENT_H_ */
