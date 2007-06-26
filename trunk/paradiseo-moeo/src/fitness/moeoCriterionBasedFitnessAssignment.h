// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoCriterionBasedFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCRITERIONBASEDFITNESSASSIGNMENT_H_
#define MOEOCRITERIONBASEDFITNESSASSIGNMENT_H_

#include <fitness/moeoFitnessAssignment.h>

/**
 * moeoCriterionBasedFitnessAssignment is a moeoFitnessAssignment for criterion-based strategies.
 */
template < class MOEOT >
class moeoCriterionBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT > {};

#endif /*MOEOCRITERIONBASEDFITNESSASSIGNMENT_H_*/
