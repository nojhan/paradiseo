// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOMETRIC_H_
#define MOEOMETRIC_H_

#include <eoFunctor.h>

/**
 * Base class for performance metrics (also known as quality indicators).
 */
class moeoMetric : public eoFunctorBase
{};


/**
 * Base class for unary metrics.
 */
template < class A, class R >
class moeoUnaryMetric : public eoUF < A, R >, public moeoMetric
{};


/**
 * Base class for binary metrics.
 */
template < class A1, class A2, class R >
class moeoBinaryMetric : public eoBF < A1, A2, R >, public moeoMetric
{};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a single solution's objective vector.
 */
template < class ObjectiveVector, class R >
class moeoSolutionUnaryMetric : public moeoUnaryMetric < const ObjectiveVector &, R >
{};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a Pareto set (a vector of objective vectors)
 */
template < class ObjectiveVector, class R >
class moeoVectorUnaryMetric : public moeoUnaryMetric < const std::vector < ObjectiveVector > &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between two solutions's objective vectors.
 */
template < class ObjectiveVector, class R >
class moeoSolutionVsSolutionBinaryMetric : public moeoBinaryMetric < const ObjectiveVector &, const ObjectiveVector &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between two Pareto sets (two vectors of objective vectors)
 */
template < class ObjectiveVector, class R >
class moeoVectorVsVectorBinaryMetric : public moeoBinaryMetric < const std::vector < ObjectiveVector > &, const std::vector < ObjectiveVector > &, R >
{};


#endif /*MOEOMETRIC_H_*/
