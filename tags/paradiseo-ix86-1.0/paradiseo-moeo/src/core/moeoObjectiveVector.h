// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTOR_H_
#define MOEOOBJECTIVEVECTOR_H_

#include <vector>

/**
 * Abstract class allowing to represent a solution in the objective space (phenotypic representation).
 * The template argument ObjectiveVectorTraits defaults to moeoObjectiveVectorTraits,
 * but it can be replaced at will by any other class that implements the static functions defined therein.
 * Some static funtions to access to the traits characteristics are re-defined in order not to write a lot of typedef's.
 */
template < class ObjectiveVectorTraits, class ObjectiveVectorType >
class moeoObjectiveVector : public std::vector < ObjectiveVectorType >
{
public:

    /** The traits of objective vectors */
    typedef ObjectiveVectorTraits Traits;
    /** The type of an objective value */
    typedef ObjectiveVectorType Type;


    /**
     * Ctor
     */
    moeoObjectiveVector(Type _value = Type()) : std::vector < Type > (ObjectiveVectorTraits::nObjectives(), _value)
    {}


    /**
     * Ctor from a vector of Type
     * @param _v the std::vector < Type >
     */
    moeoObjectiveVector(std::vector < Type > & _v) : std::vector < Type > (_v)
    {}


    /**
     * Parameters setting (for the objective vector of any solution)
     * @param _nObjectives the number of objectives
     * @param _bObjectives the min/max vector (true = min / false = max)
     */
    static void setup(unsigned int _nObjectives, std::vector < bool > & _bObjectives)
    {
        ObjectiveVectorTraits::setup(_nObjectives, _bObjectives);
    }


    /**
     * Returns the number of objectives
     */
    static unsigned int nObjectives()
    {
        return ObjectiveVectorTraits::nObjectives();
    }


    /**
     * Returns true if the _ith objective have to be minimized
     * @param _i  the index
     */
    static bool minimizing(unsigned int _i)
    {
        return ObjectiveVectorTraits::minimizing(_i);
    }


    /**
     * Returns true if the _ith objective have to be maximized
     * @param _i  the index
     */
    static bool maximizing(unsigned int _i)
    {
        return ObjectiveVectorTraits::maximizing(_i);
    }

};

#endif /*MOEOOBJECTIVEVECTOR_H_*/
