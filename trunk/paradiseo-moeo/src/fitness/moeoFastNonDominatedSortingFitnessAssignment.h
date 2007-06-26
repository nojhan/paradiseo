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

#include <vector>
#include <eoPop.h>
#include <comparator/moeoObjectiveObjectiveVectorComparator.h>
#include <comparator/moeoObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <fitness/moeoParetoBasedFitnessAssignment.h>


/**
 * Fitness assignment sheme based on Pareto-dominance count proposed in:
 * N. Srinivas, K. Deb, "Multiobjective Optimization Using Nondominated Sorting in Genetic Algorithms", Evolutionary Computation vol. 2, no. 3, pp. 221-248 (1994)
 * and in:
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * This strategy is, for instance, used in NSGA and NSGA-II.
 */
template < class MOEOT >
class moeoFastNonDominatedSortingFitnessAssignment : public moeoParetoBasedFitnessAssignment < MOEOT >
{
public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor
     */
    moeoFastNonDominatedSortingFitnessAssignment() : comparator(paretoComparator)
    {}


    /**
     * Ctor where you can choose your own way to compare objective vectors
     * @param _comparator the functor used to compare objective vectors
     */
    moeoFastNonDominatedSortingFitnessAssignment(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator) : comparator(_comparator)
    {}


    /**
     * Sets the fitness values for every solution contained in the population _pop
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
        // number of objectives for the problem under consideration
        unsigned int nObjectives = MOEOT::ObjectiveVector::nObjectives();
        if (nObjectives == 1)
        {
            // one objective
            oneObjective(_pop);
        }
        else if (nObjectives == 2)
        {
            // two objectives (the two objectives function is still to implement)
            mObjectives(_pop);
        }
        else if (nObjectives > 2)
        {
            // more than two objectives
            mObjectives(_pop);
        }
        else
        {
            // problem with the number of objectives
            throw std::runtime_error("Problem with the number of objectives in moeoNonDominatedSortingFitnessAssignment");
        }
        // a higher fitness is better, so the values need to be inverted
        double max = _pop[0].fitness();
        for (unsigned int i=1 ; i<_pop.size() ; i++)
        {
            max = std::max(max, _pop[i].fitness());
        }
        for (unsigned int i=0 ; i<_pop.size() ; i++)
        {
            _pop[i].fitness(max - _pop[i].fitness());
        }
    }


    /**
     * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            // if _pop[i] is dominated by _objVec
            if ( comparator(_pop[i].objectiveVector(), _objVec) )
            {
                _pop[i].fitness(_pop[i].fitness()+1);
            }
        }
    }


private:

    /** Functor to compare two objective vectors */
    moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;
    /** Functor allowing to compare two solutions according to their first objective value, then their second, and so on. */
    class ObjectiveComparator : public moeoComparator < MOEOT >
    {
    public:
    	/**
    	 * Returns true if _moeo1 < _moeo2 on the first objective, then on the second, and so on
    	 * @param _moeo1 the first solution
    	 * @param _moeo2 the second solution
    	 */
    	 const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    	 {
    	 	return cmp(_moeo1.objectiveVector(), _moeo2.objectiveVector());
    	 }
    private:
    	/** the corresponding comparator for objective vectors */
    	moeoObjectiveObjectiveVectorComparator < ObjectiveVector > cmp;
    } objComparator;


    /**
     * Sets the fitness values for mono-objective problems
     * @param _pop the population
     */
    void oneObjective (eoPop < MOEOT > & _pop)
    {
        // sorts the population in the ascending order
        std::sort(_pop.begin(), _pop.end(), objComparator);
        // assign fitness values
        unsigned int rank = 1;
        _pop[_pop.size()-1].fitness(rank);
        for (unsigned int i=_pop.size()-2; i>=0; i--)
        {
            if (_pop[i].objectiveVector() != _pop[i+1].objectiveVector())
            {
                rank++;
            }
            _pop[i].fitness(rank);
        }
    }


    /**
     * Sets the fitness values for bi-objective problems with a complexity of O(n log n), where n stands for the population size
     * @param _pop the population
     */
    void twoObjectives (eoPop < MOEOT > & _pop)
    {
        //... TO DO !
    }


    /**
     * Sets the fitness values for problems with more than two objectives with a complexity of O(n² log n), where n stands for the population size
     * @param _pop the population
     */
    void mObjectives (eoPop < MOEOT > & _pop)
    {
        // S[i] = indexes of the individuals dominated by _pop[i]
        std::vector < std::vector<unsigned int> > S(_pop.size());
        // n[i] = number of individuals that dominate the individual _pop[i]
        std::vector < unsigned int > n(_pop.size(), 0);
        // fronts: F[i] = indexes of the individuals contained in the ith front
        std::vector < std::vector<unsigned int> > F(_pop.size()+2);
        // used to store the number of the first front
        F[1].reserve(_pop.size());
        for (unsigned int p=0; p<_pop.size(); p++)
        {
            for (unsigned int q=0; q<_pop.size(); q++)
            {
                // if q is dominated by p
                if ( comparator(_pop[q].objectiveVector(), _pop[p].objectiveVector()) )
                {
                    // add q to the set of solutions dominated by p
                    S[p].push_back(q);
                }
                // if p is dominated by q
                else if  ( comparator(_pop[p].objectiveVector(), _pop[q].objectiveVector()) )
                {
                    // increment the domination counter of p
                    n[p]++;
                }
            }
            // if no individual dominates p
            if (n[p] == 0)
            {
                // p belongs to the first front
                _pop[p].fitness(1);
                F[1].push_back(p);
            }
        }
        // front counter
        unsigned int counter=1;
        unsigned int p,q;
        while (! F[counter].empty())
        {
            // used to store the number of the next front
            F[counter+1].reserve(_pop.size());
            for (unsigned int i=0; i<F[counter].size(); i++)
            {
                p = F[counter][i];
                for (unsigned int j=0; j<S[p].size(); j++)
                {
                    q = S[p][j];
                    n[q]--;
                    // if no individual dominates q anymore
                    if (n[q] == 0)
                    {
                        // q belongs to the next front
                        _pop[q].fitness(counter+1);
                        F[counter+1].push_back(q);
                    }
                }
            }
            counter++;
        }
    }

};

#endif /*MOEOFASTNONDOMINATEDSORTINGFITNESSASSIGNMENT_H_*/
