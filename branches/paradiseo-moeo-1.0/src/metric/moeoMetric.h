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
 * Base class for performance metrics (also called quality indicators)
 */
class moeoMetric : public eoFunctorBase
{};


/**
 * Base class for unary metrics
 */
template < class A, class R >
class moeoUnaryMetric : public eoUF < A, R >, public moeoMetric
{};


/**
 * Base class for binary metrics
 */
template < class A1, class A2, class R >
class moeoBinaryMetric : public eoBF < A1, A2, R >, public moeoMetric
{};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a single solution's Pareto fitness
 */
template < class MOEOT, class R>//, class ObjVector = typename MOEOT::ObjectiveVector >
//class moeoSolutionUnaryMetric : public moeoUnaryMetric < const ObjVector &, R >
class moeoSolutionUnaryMetric : public moeoUnaryMetric < const MOEOT &, R >
{};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a Pareto set (a vector of Pareto fitnesses)
 */
template < class MOEOT, class R>//, class ObjVector = typename MOEOT::ObjectiveVector >
//class moeoVectorUnaryMetric : public moeoUnaryMetric < const std::vector < ObjVector > &, R >
class moeoPopUnaryMetric : public moeoUnaryMetric < const eoPop < MOEOT > &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between two solutions's Pareto fitnesses
 */
template < class MOEOT, class R>//, class ObjVector = typename MOEOT::ObjectiveVector >
//class moeoSolutionVsSolutionBinaryMetric : public moeoBinaryMetric < const ObjVector &, const ObjVector &, R >
class moeoSolutionVsSolutionBinaryMetric : public moeoBinaryMetric < const MOEOT &, const MOEOT &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between a Pareto set (a vector of Pareto fitnesses) and a single solution's Pareto fitness
 */
template < class MOEOT, class R>//, class ObjVector = typename MOEOT::ObjectiveVector >
//class moeoVectorVsSolutionBinaryMetric : public moeoBinaryMetric < const std::vector < ObjVector > &, const ObjVector &, R >
class moeoPopVsSolutionBinaryMetric : public moeoBinaryMetric < const eoPop < MOEOT > &, const MOEOT &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between two Pareto sets (two vectors of Pareto fitnesses)
 */
template < class MOEOT, class R >//, class ObjVector = typename MOEOT::ObjectiveVector >
//class moeoVectorVsVectorBinaryMetric : public moeoBinaryMetric < const std::vector < ObjVector > &, const std::vector < ObjVector > &, R >
class moeoPopVsPopBinaryMetric : public moeoBinaryMetric < const eoPop < MOEOT > &, const eoPop < MOEOT > &, R >
{};


#endif /*MOEOMETRIC_H_*/
