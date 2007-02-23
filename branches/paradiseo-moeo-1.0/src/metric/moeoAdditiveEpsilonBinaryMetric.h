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
 * Additive epsilon binary metric allowing to compare two objective vectors as proposed in
 * Zitzler E., Thiele L., Laumanns M., Fonseca C. M., Grunert da Fonseca V.: 
 * Performance Assessment of Multiobjective Optimizers: An Analysis and Review. IEEE Transactions on Evolutionary Computation 7(2), pp.117–132 (2003).
 */
template < class ObjectiveVector >
class moeoAdditiveEpsilonBinaryMetric : public moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:
			
	/**
	 * Returns the maximum epsilon value by which the objective vector _o1 must be translated in all objectives 
	 * so that it weakly dominates the objective vector _o2	
	 * @warning don't forget to set the bounds for every objective before the call of this function
	 * @param _o1 the first objective vector
	 * @param _o2 the second objective vector
	 */
	double operator()(const ObjectiveVector & _o1, const ObjectiveVector & _o2)
	{
		// computation of the epsilon value for the first objective
		double result = epsilon(_o1, _o2, 0);
		// computation of the epsilon value for the other objectives
		double tmp;
		for (unsigned i=1; i<ObjectiveVector::Traits::nObjectives(); i++)
		{
			tmp = epsilon(_o1, _o2, i);
			result = std::max(result, tmp);
		}
		// returns the maximum epsilon value
    	return result;
	}


private:

	/** the bounds for every objective */
	using moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > :: bounds;
	
	
	/**
	 * Returns the epsilon value by which the objective vector _o1 must be translated in the objective _obj 
	 * so that it dominates the objective vector _o2
	 * @param _o1 the first objective vector
	 * @param _o2 the second objective vector
	 * @param _obj the index of the objective
	 */
	double epsilon(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned _obj)
	{
		double result;
		// if the objective _obj have to be minimized
		if (ObjectiveVector::Traits::minimizing(_obj))
		{
			// _o1[_obj] - _o2[_obj]
			result = ( (_o1[_obj] - bounds[_obj].minimum()) / bounds[_obj].range() ) - ( (_o2[_obj] - bounds[_obj].minimum()) / bounds[_obj].range() );
		}
		// if the objective _obj have to be maximized
		else
		{
			// _o2[_obj] - _o1[_obj]
			result = ( (_o2[_obj] - bounds[_obj].minimum()) / bounds[_obj].range() ) - ( (_o1[_obj] - bounds[_obj].minimum()) / bounds[_obj].range() );
		}
		return result;
	}

};

#endif /*MOEOADDITIVEEPSILONBINARYMETRIC_H_*/
