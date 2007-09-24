// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOINDICATORBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoFitnessAssignment.h>

/**
 * moeoIndicatorBasedFitnessAssignment is a moeoFitnessAssignment for Indicator-based strategies.
 */
template < class MOEOT >
class moeoIndicatorBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT > {};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
