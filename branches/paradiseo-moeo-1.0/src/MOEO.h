// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// MOEO.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEO_H_
#define MOEO_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <EO.h>

/**
 * Base class allowing to represent a solution (an individual) for multi-objective optimization.
 * The template argument MOEOObjectiveVector allows to represent the solution in the objective space (it can be a moeoObjectiveVector object).
 * The template argument MOEOFitness is an object reflecting the quality of the solution in term of convergence (the fitness of a solution is always to be maximized).
 * The template argument MOEODiversity is an object reflecting the quality of the solution in term of diversity (the diversity of a solution is always to be maximized).
 * All template arguments must have a void and a copy constructor.
 * Besides, note that, contrary to the mono-objective case (and to EO) where the fitness value of a solution is confused with its objective value,
 * the fitness value differs of the objectives values in the multi-objective case.
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity >
class MOEO : public EO < MOEOObjectiveVector >
{
public:

    /** the objective vector type of a solution */
    typedef MOEOObjectiveVector ObjectiveVector;

    /** the fitness type of a solution */
    typedef MOEOFitness Fitness;

    /** the diversity type of a solution */
    typedef MOEODiversity Diversity;


    /**
     * Ctor
     */
    MOEO()
    {
        // default values for every parameters
        objectiveVectorValue = ObjectiveVector();
        fitnessValue = Fitness();
        diversityValue = Diversity();
        // invalidate all
        invalidate();
    }


    /**
     * Virtual dtor
     */
    virtual ~MOEO() {};


    /**
     * Returns the objective vector of the current solution
     */
    ObjectiveVector objectiveVector() const
    {
        if ( invalidObjectiveVector() )
        {
            throw std::runtime_error("invalid objective vector");
        }
        return objectiveVectorValue;
    }


    /**
     * Sets the objective vector of the current solution
     * @param _objectiveVectorValue the new objective vector
     */
    void objectiveVector(const ObjectiveVector & _objectiveVectorValue)
    {
        objectiveVectorValue = _objectiveVectorValue;
        invalidObjectiveVectorValue = false;
    }


    /**
     * Sets the objective vector as invalid
     */
    void invalidateObjectiveVector()
    {
        invalidObjectiveVectorValue = true;
    }


    /**
     * Returns true if the objective vector is invalid, false otherwise
     */
    bool invalidObjectiveVector() const
    {
        return invalidObjectiveVectorValue;
    }


    /**
     * Returns the fitness value of the current solution
     */
    Fitness fitness() const
    {
        if ( invalidFitness() )
        {
            throw std::runtime_error("invalid fitness (MOEO)");
        }
        return fitnessValue;
    }


    /**
     * Sets the fitness value of the current solution
     * @param _fitnessValue the new fitness value
     */
    void fitness(const Fitness & _fitnessValue)
    {
        fitnessValue = _fitnessValue;
        invalidFitnessValue = false;
    }


    /**
     * Sets the fitness value as invalid
     */
    void invalidateFitness()
    {
        invalidFitnessValue = true;
    }


    /**
     * Returns true if the fitness value is invalid, false otherwise
     */
    bool invalidFitness() const
    {
        return invalidFitnessValue;
    }


    /**
     * Returns the diversity value of the current solution
     */
    Diversity diversity() const
    {
        if ( invalidDiversity() )
        {
            throw std::runtime_error("invalid diversity");
        }
        return diversityValue;
    }


    /**
     * Sets the diversity value of the current solution
     * @param _diversityValue the new diversity value
     */
    void diversity(const Diversity & _diversityValue)
    {
        diversityValue = _diversityValue;
        invalidDiversityValue = false;
    }


    /**
     * Sets the diversity value as invalid
     */
    void invalidateDiversity()
    {
        invalidDiversityValue = true;
    }


    /**
     * Returns true if the diversity value is invalid, false otherwise
     */
    bool invalidDiversity() const
    {
        return invalidDiversityValue;
    }


    /**
     * Sets the objective vector, the fitness value and the diversity value as invalid
     */
    void invalidate()
    {
        invalidateObjectiveVector();
        invalidateFitness();
        invalidateDiversity();
    }


    /**
     * Returns true if the fitness value is invalid, false otherwise
     */
    bool invalid() const
    {
        return invalidObjectiveVector();
    }


    /**
     * Returns true if the objective vector of the current solution is smaller than the objective vector of _other on the first objective, 
     * then on the second, and so on (can be usefull for sorting/printing).
     * You should implement another function in the sub-class of MOEO to have another sorting mecanism.
     * @param _other the other MOEO object to compare with
     */
    bool operator<(const MOEO & _other) const
    {
        return objectiveVector() < _other.objectiveVector();
    }


    /**
     * Return the class id (the class name as a std::string)
     */
    virtual std::string className() const
    {
        return "MOEO";
    }


    /**
     * Writing object
     * @param _os output stream
     */
    virtual void printOn(std::ostream & _os) const
    {
        if ( invalidObjectiveVector() )
        {
            _os << "INVALID\t";
        }
        else
        {
            _os << objectiveVectorValue << '\t';
        }
    }


    /**
     * Reading object
     * @param _is input stream
     */
    virtual void readFrom(std::istream & _is)
    {
        std::string objectiveVector_str;
        int pos = _is.tellg();
        _is >> objectiveVector_str;
        if (objectiveVector_str == "INVALID")
        {
            invalidateObjectiveVector();
        }
        else
        {
            invalidObjectiveVectorValue = false;
            _is.seekg(pos); // rewind
            _is >> objectiveVectorValue;
        }
    }


private:

    /** the objective vector of this solution */
    ObjectiveVector objectiveVectorValue;
    /** true if the objective vector is invalid */
    bool invalidObjectiveVectorValue;
    /** the fitness value of this solution */
    Fitness fitnessValue;
    /** true if the fitness value is invalid */
    bool invalidFitnessValue;
    /** the diversity value of this solution */
    Diversity diversityValue;
    /** true if the diversity value is invalid */
    bool invalidDiversityValue;

};

#endif /*MOEO_H_*/
