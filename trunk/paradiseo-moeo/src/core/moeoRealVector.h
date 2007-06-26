// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoRealVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOREALVECTOR_H_
#define MOEOREALVECTOR_H_

#include <core/moeoVector.h>

/**
 * This class is an implementation of a simple double-valued moeoVector.
 */
template < class MOEOObjectiveVector, class MOEOFitness, class MOEODiversity >
class moeoRealVector : public moeoVector < MOEOObjectiveVector, MOEOFitness, MOEODiversity, double >
{
public:

    /**
     * Ctor
     * @param _size Length of vector (default is 0)
     * @param _value Initial value of all elements (default is default value of type GeneType)
     */
    moeoRealVector(unsigned int _size = 0, double _value = 0.0) : moeoVector< MOEOObjectiveVector, MOEOFitness, MOEODiversity, double >(_size, _value)
    {}

};

#endif /*MOEOREALVECTOR_H_*/
