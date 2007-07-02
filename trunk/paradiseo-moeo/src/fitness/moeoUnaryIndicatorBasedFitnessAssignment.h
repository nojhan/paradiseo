// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoUnaryIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOUNARYINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOUNARYINDICATORBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoIndicatorBasedFitnessAssignment.h>

/**
 * moeoIndicatorBasedFitnessAssignment for unary indicators.
 */
template < class MOEOT >
class moeoUnaryIndicatorBasedFitnessAssignment : public moeoIndicatorBasedFitnessAssignment < MOEOT > {};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
