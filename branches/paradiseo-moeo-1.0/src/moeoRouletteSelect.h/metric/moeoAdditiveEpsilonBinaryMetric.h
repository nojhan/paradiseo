// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoAdditiveEpsilonBinaryMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOADDITIVEEPSILONBINARYMETRIC_H_
#define MOEOADDITIVEEPSILONBINARYMETRIC_H_

#include <metric/moeoMetric.h>

/**
 * 
 */
template < class MOEOT >
class moeoAdditiveEpsilonBinaryMetric : public moeoSolutionVsSolutionBinaryMetric < MOEOT, double >
{
public:
	
	/** the objective vector type of a solution */
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
	moeoAdditiveEpsilonBinaryMetric();
	
	
	double operator()(const std::vector < ObjectiveVector > & _set1, const std::vector < ObjectiveVector > & _set2)
	{
		
	}
	
};

#endif /*MOEOADDITIVEEPSILONBINARYMETRIC_H_*/
