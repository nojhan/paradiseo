// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoFitnessAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOFITNESSASSIGNMENT_H_
#define MOEOFITNESSASSIGNMENT_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor that sets the fitness values of a whole population.
 */
template < class MOEOT >
class moeoFitnessAssignment : public eoUF < eoPop < MOEOT > &, void >
{};


/**
 * moeoScalarFitnessAssignment is a moeoFitnessAssignment for scalar strategies.
 */
template < class MOEOT >
class moeoScalarFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


/**
 * moeoCriterionBasedFitnessAssignment is a moeoFitnessAssignment for criterion-based strategies.
 */
template < class MOEOT >
class moeoCriterionBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


/**
 * moeoParetoBasedFitnessAssignment is a moeoFitnessAssignment for Pareto-based strategies.
 */
template < class MOEOT >
class moeoParetoBasedFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{};


#endif /*MOEOFITNESSASSIGNMENT_H_*/
