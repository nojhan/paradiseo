// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoRealObjectiveVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOREALOBJECTIVEVECTOR_H_
#define MOEOREALOBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>
#include <comparator/moeoObjectiveObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <core/moeoObjectiveVector.h>

/**
 * This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of real values,
 * i.e. that an objective value is represented using a double, and this for any objective.
 */
template < class ObjectiveVectorTraits >
class moeoRealObjectiveVector : public moeoObjectiveVector < ObjectiveVectorTraits, double >
{
public:

    using moeoObjectiveVector < ObjectiveVectorTraits, double >::size;
    using moeoObjectiveVector < ObjectiveVectorTraits, double >::operator[];

    /**
     * Ctor
     */
    moeoRealObjectiveVector(double _value = 0.0) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_value)
    {}


    /**
     * Ctor from a vector of doubles
     * @param _v the std::vector < double >
     */
    moeoRealObjectiveVector(std::vector < double > & _v) : moeoObjectiveVector < ObjectiveVectorTraits, double > (_v)
    {}


    /**
     * Returns true if the current objective vector dominates _other according to the Pareto dominance relation
     * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool dominates(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        moeoParetoObjectiveVectorComparator < moeoRealObjectiveVector<ObjectiveVectorTraits> > comparator;
        return comparator(_other, *this);
    }


    /**
     * Returns true if the current objective vector is equal to _other (according to a tolerance value)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator==(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
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
     * @param _other the other moeoRealObjectiveVector object to compare with 
     */
    bool operator!=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        return ! operator==(_other);
    }


    /**
     * Returns true if the current objective vector is smaller than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator<(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        moeoObjectiveObjectiveVectorComparator < moeoRealObjectiveVector < ObjectiveVectorTraits > > cmp;
        return cmp(*this, _other);
    }


    /**
     * Returns true if the current objective vector is greater than _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator>(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        return _other < *this;
    }


    /**
     * Returns true if the current objective vector is smaller than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator<=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        return operator==(_other) || operator<(_other);
    }


    /**
     * Returns true if the current objective vector is greater than or equal to _other on the first objective, then on the second, and so on
     * (can be usefull for sorting/printing)
     * @param _other the other moeoRealObjectiveVector object to compare with
     */
    bool operator>=(const moeoRealObjectiveVector < ObjectiveVectorTraits > & _other) const
    {
        return operator==(_other) || operator>(_other);
    }

};


/**
 * Output for a moeoRealObjectiveVector object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */
template < class ObjectiveVectorTraits >
std::ostream & operator<<(std::ostream & _os, const moeoRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
    for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
        _os << _objectiveVector[i] << '\t';
    }
    return _os;
}

/**
 * Input for a moeoRealObjectiveVector object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits >
std::istream & operator>>(std::istream & _is, moeoRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
    _objectiveVector = moeoRealObjectiveVector < ObjectiveVectorTraits > ();
    for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
        _is >> _objectiveVector[i];
    }
    return _is;
}

#endif /*MOEOREALOBJECTIVEVECTOR_H_*/
