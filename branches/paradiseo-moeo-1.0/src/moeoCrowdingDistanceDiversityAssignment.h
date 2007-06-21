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
        * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
        */
    double tiny() const
    {
        return 1e-6;
    }


    /**
     * Computes diversity values for every solution contained in the population _pop
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
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


protected:

    /**
     * Sets the distance values
     * @param _pop the population
     */
    virtual void setDistances (eoPop < MOEOT > & _pop)
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


/**
 * Diversity assignment sheme based on crowding distance proposed in:
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * Tis strategy assigns diversity values FRONT BY FRONT. It is, for instance, used in NSGA-II.
 */
template < class MOEOT >
class moeoFrontByFrontCrowdingDistanceDiversityAssignment : public moeoCrowdingDistanceDiversityAssignment < MOEOT >
{
public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * @warning NOT IMPLEMENTED, DO NOTHING !
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     * @warning NOT IMPLEMENTED, DO NOTHING !
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        cout << "WARNING : updateByDeleting not implemented in moeoFrontByFrontCrowdingDistanceDiversityAssignment" << endl;
    }


private:

    using moeoCrowdingDistanceDiversityAssignment < MOEOT >::inf;
    using moeoCrowdingDistanceDiversityAssignment < MOEOT >::tiny;


    /**
     * Sets the distance values
     * @param _pop the population
     */
    void setDistances (eoPop < MOEOT > & _pop)
    {
        unsigned a,b;
        double min, max, distance;
        unsigned nObjectives = MOEOT::ObjectiveVector::nObjectives();
        // set diversity to 0 for every individual
        for (unsigned i=0; i<_pop.size(); i++)
        {
            _pop[i].diversity(0.0);
        }
        // sort the whole pop according to fitness values
        moeoFitnessThenDiversityComparator < MOEOT > fitnessComparator;
        std::sort(_pop.begin(), _pop.end(), fitnessComparator);
        // compute the crowding distance values for every individual "front" by "front" (front : from a to b)
        a = 0;	        			// the front starts at a
        while (a < _pop.size())
        {
            b = lastIndex(_pop,a);	// the front ends at b
            // if there is less than 2 individuals in the front...
            if ((b-a) < 2)
            {
                for (unsigned i=a; i<=b; i++)
                {
                    _pop[i].diversity(inf());
                }
            }
            // else...
            else
            {
                // for each objective
                for (unsigned obj=0; obj<nObjectives; obj++)
                {
                    // sort in the descending order using the values of the objective 'obj'
                    moeoOneObjectiveComparator < MOEOT > objComp(obj);
                    std::sort(_pop.begin()+a, _pop.begin()+b+1, objComp);
                    // min & max
                    min = _pop[b].objectiveVector()[obj];
                    max = _pop[a].objectiveVector()[obj];
                    // avoid extreme case
                    if (min == max)
                    {
                        min -= tiny();
                        max += tiny();
                    }
                    // set the diversity value to infiny for min and max
                    _pop[a].diversity(inf());
                    _pop[b].diversity(inf());
                    // set the diversity values for the other individuals
                    for (unsigned i=a+1; i<b; i++)
                    {
                        distance = (_pop[i-1].objectiveVector()[obj] - _pop[i+1].objectiveVector()[obj]) / (max-min);
                        _pop[i].diversity(_pop[i].diversity() + distance);
                    }
                }
            }
            // go to the next front
            a = b+1;
        }
    }


    /**
     * Returns the index of the last individual having the same fitness value than _pop[_start]
     * @param _pop the population
     * @param _start the index to start from
     */
    unsigned lastIndex (eoPop < MOEOT > & _pop, unsigned _start)
    {
        unsigned i=_start;
        while ( (i<_pop.size()-1) && (_pop[i].fitness()==_pop[i+1].fitness()) )
        {
            i++;
        }
        return i;
    }

};

#endif /*MOEOCROWDINGDISTANCEDIVERSITYASSIGNMENT_H_*/
