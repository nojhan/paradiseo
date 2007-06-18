// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoAchievementFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOACHIEVEMENTFITNESSASSIGNMENT_H_
#define MOEOACHIEVEMENTFITNESSASSIGNMENT_H_

#include <eoPop.h>
#include <moeoFitnessAssignment.h>

/**
 *
 */
template < class MOEOT >
class moeoAchievementFitnessAssignment : public moeoScalarFitnessAssignment < MOEOT >
{
public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
        * Default ctor
        * @param _reference reference point vector
        * @param _lambdas weighted coefficients vector
        * @param _spn arbitrary small positive number (0 < _spn << 1)
        */
    moeoAchievementFitnessAssignment(ObjectiveVector & _reference, vector < double > & _lambdas, double _spn=0.0001) : reference(_reference), lambdas(_lambdas), spn(_spn)
    {
        // consistency check
        if ((spn < 0.0) || (spn > 1.0))
        {
            std::cout << "Warning, the arbitrary small positive number should be > 0 and <<1, adjusted to 0.0001\n";
            spn = 0.0001;
        }
    }


    /**
        * Ctor with default values for lambdas (1/nObjectives)
        * @param _reference reference point vector
        * @param _spn arbitrary small positive number (0 < _spn << 1)
        */
    moeoAchievementFitnessAssignment(ObjectiveVector & _reference, double _spn=0.0001) : reference(_reference), spn(_spn)
    {
        // compute the default values for lambdas
        lambdas = vector < double > (ObjectiveVector::nObjectives());
        for (unsigned i=0 ; i<lambdas.size(); i++)
        {
            lambdas[i] = 1.0 / ObjectiveVector::nObjectives();
        }
        // consistency check
        if ((spn < 0.0) || (spn > 1.0))
        {
            std::cout << "Warning, the arbitrary small positive number should be > 0 and <<1, adjusted to 0.0001\n";
            spn = 0.0001;
        }
    }


    /**
     * Sets the fitness values for every solution contained in the population _pop
        * @param _pop the population
     */
    virtual void operator()(eoPop < MOEOT > & _pop)
    {
        for (unsigned i=0; i<_pop.size() ; i++)
        {
            compute(_pop[i]);
        }
    }


    /**
        * Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account (nothing to do).
        * @param _pop the population
        * @param _objVec the objective vector
        */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        // nothing to do ;-)
    }


    /**
     * Sets the reference point
     * @param _reference the new reference point
     */
    void setReference(const ObjectiveVector & _reference)
    {
        reference = _reference;
    }


private:

    /** the reference point */
    ObjectiveVector reference;
    /** the weighted coefficients vector */
    vector < double > lambdas;
    /** an arbitrary small positive number (0 < _spn << 1) */
    double spn;


    /**
     * Returns a big value (regarded as infinite)
     */
    double inf() const
    {
        return std::numeric_limits<double>::max();
    }


    /**
     * Computes the fitness value for a solution
     * @param _moeo the solution
     */
    void compute(MOEOT & _moeo)
    {
        unsigned nobj = MOEOT::ObjectiveVector::nObjectives();
        double temp;
        double min = inf();
        double sum = 0;
        for (unsigned obj=0; obj<nobj; obj++)
        {
            temp = lambdas[obj] * (reference[obj] - _moeo.objectiveVector()[obj]);
            min = std::min(min, temp);
            sum += temp;
        }
        _moeo.fitness(min + spn*sum);
    }

};

#endif /*MOEOACHIEVEMENTFITNESSASSIGNMENT_H_*/
