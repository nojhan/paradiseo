// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoNormalizedSolutionVsSolutionBinaryMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEONORMALIZEDSOLUTIONVSSOLUTIONBINARYMETRIC_H_
#define MOEONORMALIZEDSOLUTIONVSSOLUTIONBINARYMETRIC_H_

#include <stdexcept>
#include <metric/moeoMetric.h>


/**
 * Base class for binary metrics dedicated to the performance comparison between two solutions's objective vectors using normalized values.
 * Then, indicator values lie in the interval [-1,1].
 * Note that you have to set the bounds for every objective before using the operator().
 */
template < class ObjectiveVector, class R >
class moeoNormalizedSolutionVsSolutionBinaryMetric : public moeoSolutionVsSolutionBinaryMetric < ObjectiveVector, R >
{
public:

    /** very small value to avoid the extreme case where the min bound == the max bound */
    const static double tiny = 1e-6;


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
    void setup(double _min, double _max, unsigned _obj)
    {
        if (_min == _max)
        {
            _min -= tiny;
            _max += tiny;
        }
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
 * Additive epsilon binary metric allowing to compare two objective vectors as proposed in
 * Zitzler E., Thiele L., Laumanns M., Fonseca C. M., Grunert da Fonseca V.:
 * Performance Assessment of Multiobjective Optimizers: An Analysis and Review. IEEE Transactions on Evolutionary Computation 7(2), pp.117–132 (2003).
 */
template < class ObjectiveVector >
class moeoAdditiveEpsilonBinaryMetric : public moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:

    /**
     * Returns the minimal distance by which the objective vector _o1 must be translated in all objectives 
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


/**
 * Hypervolume binary metric allowing to compare two objective vectors as proposed in
 * Zitzler E., Künzli S.: Indicator-Based Selection in Multiobjective Search. In Parallel Problem Solving from Nature (PPSN VIII).
 * Lecture Notes in Computer Science 3242, Springer, Birmingham, UK pp.832–842 (2004).
 * This indicator is based on the hypervolume concept introduced in
 * Zitzler, E., Thiele, L.: Multiobjective Optimization Using Evolutionary Algorithms - A Comparative Case Study.
 * Parallel Problem Solving from Nature (PPSN-V), pp.292-301 (1998).
 */
template < class ObjectiveVector >
class moeoHypervolumeBinaryMetric : public moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:

    /**
     * Ctor
     * @param _rho value used to compute the reference point from the worst values for each objective (default : 1.1)
     */
    moeoHypervolumeBinaryMetric(double _rho = 1.1) : rho(_rho)
    {
        // not-a-maximization problem check
        for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            if (ObjectiveVector::Traits::maximizing(i))
            {
                throw std::runtime_error("Hypervolume binary metric not yet implemented for a maximization problem in moeoHypervolumeBinaryMetric");
            }
        }
        // consistency check
        if (rho < 1)
        {
            cout << "Warning, value used to compute the reference point rho for the hypervolume calculation must not be smaller than 1" << endl;
            cout << "Adjusted to 1" << endl;
            rho = 1;
        }
    }


    /**
     * Returns the volume of the space that is dominated by _o2 but not by _o1 with respect to a reference point computed using rho.
     * @warning don't forget to set the bounds for every objective before the call of this function
     * @param _o1 the first objective vector
     * @param _o2 the second objective vector
     */
    double operator()(const ObjectiveVector & _o1, const ObjectiveVector & _o2)
    {
        double result;
        // if _o1 dominates _o2
        if ( paretoComparator(_o1,_o2) )
        {
            result = - hypervolume(_o1, _o2, ObjectiveVector::Traits::nObjectives()-1);
        }
        else
        {
            result = hypervolume(_o2, _o1, ObjectiveVector::Traits::nObjectives()-1);
        }
        return result;
    }


private:

    /** value used to compute the reference point from the worst values for each objective */
    double rho;
    /** the bounds for every objective */
    using moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double > :: bounds;
    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;

    /**
     * Returns the volume of the space that is dominated by _o2 but not by _o1 with respect to a reference point computed using rho for the objective _obj.
     * @param _o1 the first objective vector
     * @param _o2 the second objective vector
     * @param _obj the objective index
     * @param _flag used for iteration, if _flag=true _o2 is not talen into account (default : false)
     */
    double hypervolume(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned _obj, const bool _flag = false)
    {
        double result;
        double range = rho * bounds[_obj].range();
        double max = bounds[_obj].minimum() + range;
        // value of _1 for the objective _obj
        double v1 = _o1[_obj];
        // value of _2 for the objective _obj (if _flag=true, v2=max)
        double v2;
        if (_flag)
        {
            v2 = max;
        }
        else
        {
            v2 = _o2[_obj];
        }
        // computation of the volume
        if (_obj == 0)
        {
            if (v1 < v2)
            {
                result = (v2 - v1) / range;
            }
            else
            {
                result = 0;
            }
        }
        else
        {
            if (v1 < v2)
            {
                result = ( hypervolume(_o1, _o2, _obj-1, true) * (v2 - v1) / range ) + ( hypervolume(_o1, _o2, _obj-1) * (max - v2) / range );
            }
            else
            {
                result = hypervolume(_o1, _o2, _obj-1) * (max - v2) / range;
            }
        }
        return result;
    }

};


#endif /*MOEONORMALIZEDSOLUTIONVSSOLUTIONBINARYMETRIC_H_*/
