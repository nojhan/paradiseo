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
 * Base class for binary metrics dedicated to the performance comparison between two solutions's objective vectors using normalized values.
 * Then, indicator values lie in the interval [-1,1].
 * Note that you have to set the bounds for every objective before using the operator().
 */
template < class ObjectiveVector, class R >
class moeoNormalizedSolutionVsSolutionBinaryMetric : public moeoSolutionVsSolutionBinaryMetric < ObjectiveVector, R >
{
public:

	/**
	 * Default ctr for any moeoNormalizedSolutionVsSolutionBinaryMetric object
	 */
	moeoNormalizedSolutionVsSolutionBinaryMetric()
	{
		bounds.resize(ObjectiveVector::Traits::nObjectives());
	}
	
	
	/**
	 * Sets the lower bound (_min) and the upper bound (_max) for the objective _obj
	 * _min lower bound
	 * _max upper bound
	 * _obj the objective index
	 */
	virtual void setup(double _min, double _max, unsigned _obj)
	{
		bounds[_obj] = eoRealInterval(_min, _max);
	}
	
	/**
	 * Sets the lower bound and the upper bound for the objective _obj using a eoRealInterval object
	 * _realInterval the eoRealInterval object
	 * _obj the objective index
	 */
	virtual void setup(eoRealInterval _realInterval, unsigned _obj)
	{
		bounds[_obj] = _realInterval;
	}
	
	
protected:

	/** the bounds for every objective (bounds[i] = bounds for the objective i) */
	std::vector < eoRealInterval > bounds;
	
};


/**
 * Base class for binary metrics dedicated to the performance comparison between a Pareto set (a vector of objective vectors) and a single solution's objective vector.
 */
template < class ObjectiveVector, class R >
class moeoVectorVsSolutionBinaryMetric : public moeoBinaryMetric < const std::vector < ObjectiveVector > &, const ObjectiveVector &, R >
{};


/**
 * Base class for binary metrics dedicated to the performance comparison between two Pareto sets (two vectors of objective vectors)
 */
template < class ObjectiveVector, class R >
class moeoVectorVsVectorBinaryMetric : public moeoBinaryMetric < const std::vector < ObjectiveVector > &, const std::vector < ObjectiveVector > &, R >
{};


#endif /*MOEOMETRIC_H_*/
