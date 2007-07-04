// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoScalarFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSCALARFITNESSASSIGNMENT_H_
#define MOEOSCALARFITNESSASSIGNMENT_H_

#include <fitness/moeoFitnessAssignment.h>

/**
 * moeoScalarFitnessAssignment is a moeoFitnessAssignment for scalar strategies.
 */
template < class MOEOT >
class moeoScalarFitnessAssignment : public moeoFitnessAssignment < MOEOT > {};
    
#endif /*MOEOSCALARFITNESSASSIGNMENT_H_*/
