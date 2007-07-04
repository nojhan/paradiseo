// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoBinaryIndicatorBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOBINARYINDICATORBASEDFITNESSASSIGNMENT_H_
#define MOEOBINARYINDICATORBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoIndicatorBasedFitnessAssignment.h>

/**
 * moeoIndicatorBasedFitnessAssignment for binary indicators.
 */
template < class MOEOT >
class moeoBinaryIndicatorBasedFitnessAssignment : public moeoIndicatorBasedFitnessAssignment < MOEOT > {};

#endif /*MOEOINDICATORBASEDFITNESSASSIGNMENT_H_*/
