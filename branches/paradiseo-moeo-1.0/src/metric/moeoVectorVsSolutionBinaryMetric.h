// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoVectorVsSolutionBinaryMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOVECTORVSSOLUTIONBINARYMETRIC_H_
#define MOEOVECTORVSSOLUTIONBINARYMETRIC_H_

#include <metric/moeoMetric.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>


/**
 * Base class for binary metrics dedicated to the performance comparison between a Pareto set (a vector of objective vectors) and a single solution's objective vector.
 */
template < class ObjectiveVector, class R >
class moeoVectorVsSolutionBinaryMetric : public moeoBinaryMetric < const std::vector < ObjectiveVector > &, const ObjectiveVector &, R >
{
public:

	/**
	 * Default ctor
	 * @param _metric the binary metric for the performance comparison between two solutions's objective vectors using normalized values
	 */
	moeoVectorVsSolutionBinaryMetric(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric) : metric(_metric)
	{}
	
	/**
	 * Returns the value of the metric comparing the set _v to an objective vector _o
	 * _v a vector of objective vectors
	 * _o an objective vector
	 */
	double operator()(const std::vector < ObjectiveVector > & _v, const ObjectiveVector & _o)
	{
		// 1 - set the bounds for every objective
		setBounds(_v, _o);
		// 2 - compute every indicator value
		computeValues(_v, _o);
		// 3 - resulting value
		return result();
	}

	
protected:

	/** the binary metric for the performance comparison between two solutions's objective vectors using normalized values */
	moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * metric;
	/** the indicator values : values[i] = I(_v[i], _o) */
	vector < double > values;
	
	
	/**
	 * Sets the bounds for every objective using the min and the max value
	 * _v a vector of objective vectors
	 * _o an objective vector
	 */
	void setBounds(const std::vector < ObjectiveVector > & _v, const ObjectiveVector & _o)
	{
		double min, max;
		for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
		{
			min = _o[i];
			max = _o[i];
			for (unsigned j=0; j<_v.size(); j++)
			{
				min = std::min(min, _v[j][i]);
				max = std::max(max, _v[j][i]);
			}
			// setting of the bounds for the objective i
			(*metric).setup(min, max, i);			
		}
	}
	
	
	/**
	 * Compute every indicator value : values[i] = I(_v[i], _o)
	 * _v a vector of objective vectors
	 * _o an objective vector
	 */
	void computeValues(const std::vector < ObjectiveVector > & _v, const ObjectiveVector & _o)
	{
		values.clear();
		values.resize(_v.size());
		for (unsigned i=0; i<_v.size(); i++)
		{
			values[i] = (*metric)(_v[i], _o);
		}
	}
	
	
	/**
	 * Returns the global result that combines the I-values
	 */
	virtual double result() = 0;
	
};


/**
 * Minimum version of binary metric dedicated to the performance comparison between a vector of objective vectors and a single solution's objective vector.
 */
template < class ObjectiveVector >
class moeoMinimumVectorVsSolutionBinaryMetric : public moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:

	/**
	 * Ctor
	 * @param _metric the binary metric for the performance comparison between two solutions's objective vectors using normalized values
	 */
	moeoMinimumVectorVsSolutionBinaryMetric(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric) : moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > (_metric)
	{}


private:

	/** the indicator values : values[i] = I(_v[i], _o) */
	using moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >::values;
	
	
	/**
	 * Returns the minimum binary indicator values computed
	 */
	double result()
	{
		return *std::min_element(values.begin(), values.end());
	}

};


/**
 * Additive version of binary metric dedicated to the performance comparison between a vector of objective vectors and a single solution's objective vector.
 */
template < class ObjectiveVector >
class moeoAdditiveVectorVsSolutionBinaryMetric : public moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:

	/**
	 * Ctor
	 * @param _metric the binary metric for the performance comparison between two solutions's objective vectors using normalized values
	 */
	moeoAdditiveVectorVsSolutionBinaryMetric(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric) : moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > (_metric)
	{}


private:

	/** the indicator values : values[i] = I(_v[i], _o) */
	using moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >::values;
	
	
	/**
	 * Returns the sum of the binary indicator values computed
	 */
	double result()
	{
		double result = 0;
		for (unsigned i=0; i<values.size(); i++)
		{
			result += values[i];
		}
		return result;
	}

};


/**
 * Exponential version of binary metric dedicated to the performance comparison between a vector of objective vectors 
 * and a single solution's objective vector.
 * 
 * ********** Do we have to care about the max absolute indicator value ? ********************
 * 
 */
template < class ObjectiveVector >
class moeoExponentialVectorVsSolutionBinaryMetric : public moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:

	/**
	 * Ctor
	 * @param _metric the binary metric for the performance comparison between two solutions's objective vectors using normalized values
	 */
	moeoExponentialVectorVsSolutionBinaryMetric(moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > * _metric, const double _kappa) : 
	moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double > (_metric), kappa(_kappa)
	{}
	

private:

	/** scaling factor kappa */
	double kappa;
	/** the indicator values : values[i] = I(_v[i], _o) */
	using moeoVectorVsSolutionBinaryMetric < ObjectiveVector, double >::values;
	
	
	/**
	 * Returns a kind of sum of the binary indicator values computed that amplifies the influence of dominating objective vectors over dominated ones
	 */
	double result()
	{
		double result = 0;
		for (unsigned i=0; i<values.size(); i++)
		{
			result += exp(-values[i] / kappa);
		}
		return result;
	}

};

#endif /*MOEOVECTORVSSOLUTIONBINARYMETRIC_H_*/
