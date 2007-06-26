// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoParetoBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOPARETOBASEDFITNESSASSIGNMENT_H_
#define MOEOPARETOBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoFitnessAssignment.h>

/**
 * moeoParetoBasedFitnessAssignment is a moeoFitnessAssignment for Pareto-based strategies.
 */
template < class MOEOT >
class moeoParetoBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT > {};
    
#endif /*MOEOPARETOBASEDFITNESSASSIGNMENT_H_*/
