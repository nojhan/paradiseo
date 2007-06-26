// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDistance.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODISTANCE_H_
#define MOEODISTANCE_H_

#include <eoFunctor.h>

/**
 * The base class for distance computation.
 */
template < class MOEOT , class Type >
class moeoDistance : public eoBF < const MOEOT &, const MOEOT &, const Type >
{
public:

    /**
     * Nothing to do
     * @param _pop the population
     */
    virtual void setup(const eoPop < MOEOT > & _pop)
    {}


    /**
     * Nothing to do
     * @param _min lower bound
     * @param _max upper bound
     * @param _obj the objective index
     */
    virtual void setup(double _min, double _max, unsigned int _obj)
    {}


    /**
     * Nothing to do
     * @param _realInterval the eoRealInterval object
     * @param _obj the objective index
     */
    virtual void setup(eoRealInterval _realInterval, unsigned int _obj)
    {}

};

#endif /*MOEODISTANCE_H_*/
