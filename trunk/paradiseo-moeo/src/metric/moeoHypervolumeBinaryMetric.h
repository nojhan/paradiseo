// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoHypervolumeBinaryMetric.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOHYPERVOLUMEBINARYMETRIC_H_
#define MOEOHYPERVOLUMEBINARYMETRIC_H_

#include <stdexcept>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <metric/moeoNormalizedSolutionVsSolutionBinaryMetric.h>

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
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            if (ObjectiveVector::Traits::maximizing(i))
            {
                throw std::runtime_error("Hypervolume binary metric not yet implemented for a maximization problem in moeoHypervolumeBinaryMetric");
            }
        }
        // consistency check
        if (rho < 1)
        {
            std::cout << "Warning, value used to compute the reference point rho for the hypervolume calculation must not be smaller than 1" << std::endl;
            std::cout << "Adjusted to 1" << std::endl;
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
        // if _o2 is dominated by _o1
        if ( paretoComparator(_o2,_o1) )
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
    double hypervolume(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned int _obj, const bool _flag = false)
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

#endif /*MOEOHYPERVOLUMEBINARYMETRIC_H_*/
