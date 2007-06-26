// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVectorDouble.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTORDOUBLE_H_
#define MOEOOBJECTIVEVECTORDOUBLE_H_

#include <iostream>
#include <math.h>
#include <comparator/moeoObjectiveObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <core/moeoObjectiveVector.h>

/**
 * This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of doubles,
 * i.e. that an objective value is represented using a double, and this for any objective.
 */
template < class ObjectiveVectorTraits >
class moeoObjectiveVectorDouble : public moeoObjectiveVector < ObjectiveVectorTraits, double >
{
public:

    using moeoObjectiveVector < ObjectiveVectorTraits, double >::size;
    using moeoObjectiveVector < ObjectiveVectorTraits, double >::operator[];

    /**
     * Ctor
     */
    moeoObjectiveVectorDouble(double _value = 0.0) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_value)
    {}


    /**
     * Ctor from a vector of doubles
     * @param _v the std::vector < double >
     */
    moeoObjectiveVectorDouble(std::vector < double > & _v) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_v)
    {}


    /**
     * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
     * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool dominates(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        moeoParetoObjectiveVectorComparator < moeoObjectiveVectorDouble<ObjectiveVectorTraits> > comparator;
        return comparator(_other, *this);
    }


    /**
     * Returns true if the current objective vector is equal to _other (according to a tolerance value)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool operator==(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        for (unsigned int i=0; i < size(); i++)
        {
            if ( fabs(operator[](i) - _other[i]) > ObjectiveVectorTraits::tolerance() )
            {
                return false;
            }
        }
        return true;
    }


    /**
     * Returns true if the current objective vector is different than _other (according to a tolerance value)
     * @param _other the other moeoObjectiveVectorDouble object to compare with 
     */
    bool operator!=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        return ! operator==(_other);
    }


    /**
     * Returns true if the current objective vector is smaller than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool operator<(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        moeoObjectiveObjectiveVectorComparator < moeoObjectiveVectorDouble < ObjectiveVectorTraits > > cmp;
        return cmp(*this, _other);
    }


    /**
     * Returns true if the current objective vector is greater than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool operator>(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        return _other < *this;
    }


    /**
     * Returns true if the current objective vector is smaller than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool operator<=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        return operator==(_other) || operator<(_other);
    }


    /**
     * Returns true if the current objective vector is greater than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoObjectiveVectorDouble object to compare with
     */
    bool operator>=(const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _other) const
    {
        return operator==(_other) || operator>(_other);
    }

};


/**
 * Output for a moeoObjectiveVectorDouble object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template < class ObjectiveVectorTraits >
std::ostream & operator<<(std::ostream & _os, const moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _objectiveVector)
{
    for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
        _os << _objectiveVector[i] << '\t';
    }
    return _os;
}

/**
 * Input for a moeoObjectiveVectorDouble object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits >
std::istream & operator>>(std::istream & _is, moeoObjectiveVectorDouble < ObjectiveVectorTraits > & _objectiveVector)
{
    _objectiveVector = moeoObjectiveVectorDouble < ObjectiveVectorTraits > ();
    for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
        _is >> _objectiveVector[i];
    }
    return _is;
}

#endif /*MOEOOBJECTIVEVECTORDOUBLE_H_*/
