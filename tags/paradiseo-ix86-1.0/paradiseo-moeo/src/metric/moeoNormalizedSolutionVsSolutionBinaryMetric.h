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

#include <vector>
#include <utils/eoRealBounds.h>
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

    /**
     * Default ctr for any moeoNormalizedSolutionVsSolutionBinaryMetric object
     */
    moeoNormalizedSolutionVsSolutionBinaryMetric()
    {
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }


    /**
     * Sets the lower bound (_min) and the upper bound (_max) for the objective _obj
     * @param _min lower bound
     * @param _max upper bound
     * @param _obj the objective index
     */
    void setup(double _min, double _max, unsigned int _obj)
    {
        if (_min == _max)
        {
            _min -= tiny();
            _max += tiny();
        }
        bounds[_obj] = eoRealInterval(_min, _max);
    }


    /**
     * Sets the lower bound and the upper bound for the objective _obj using a eoRealInterval object
     * @param _realInterval the eoRealInterval object
     * @param _obj the objective index
     */
    virtual void setup(eoRealInterval _realInterval, unsigned int _obj)
    {
        bounds[_obj] = _realInterval;
    }


    /**
     * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
     */
    static double tiny()
    {
        return 1e-6;
    }


protected:

    /** the bounds for every objective (bounds[i] = bounds for the objective i) */
    std::vector < eoRealInterval > bounds;

};

#endif /*MOEONORMALIZEDSOLUTIONVSSOLUTIONBINARYMETRIC_H_*/
