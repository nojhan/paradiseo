// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoOneObjectiveComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOONEOBJECTIVECOMPARATOR_H_
#define MOEOONEOBJECTIVECOMPARATOR_H_

#include <comparator/moeoComparator.h>

/**
 * Functor allowing to compare two solutions according to one objective.
 */
template < class MOEOT >
class moeoOneObjectiveComparator : public moeoComparator < MOEOT >
{
public:

    /**
     * Ctor.
     * @param _obj the index of objective
     */
    moeoOneObjectiveComparator(unsigned int _obj) : obj(_obj)
    {
        if (obj > MOEOT::ObjectiveVector::nObjectives())
        {
            throw std::runtime_error("Problem with the index of objective in moeoOneObjectiveComparator");
        }
    }


    /**
     * Returns true if _moeo1 < _moeo2 on the obj objective
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
    const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
        return _moeo1.objectiveVector()[obj] < _moeo2.objectiveVector()[obj];
    }


private:

    /** the index of objective */
    unsigned int obj;

};

#endif /*MOEOONEOBJECTIVECOMPARATOR_H_*/
