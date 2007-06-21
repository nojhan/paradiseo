// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDistance.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODISTANCE_H_
#define MOEODISTANCE_H_

#include <math.h>
#include <eoFunctor.h>
#include <utils/eoRealBounds.h>

/**
 * The base class for distance computation.
 */
template < class MOEOT , class Type >
class moeoDistance : public eoBF < const MOEOT &, const MOEOT &, const Type >
{
public:


	/**
     * Nothing to do
     * @param _pop the population
     */
    virtual void setup(const eoPop < MOEOT > & _pop)
    {}
    
    /**
     * Nothing to do
     * @param _min lower bound
     * @param _max upper bound
     * @param _obj the objective index
     */
    virtual void setup(double _min, double _max, unsigned _obj)
    {}


	/**
     * Nothing to do
     * @param _realInterval the eoRealInterval object
     * @param _obj the objective index
     */
    virtual void setup(eoRealInterval _realInterval, unsigned _obj)
    {}
    
};


/**
 * The base class for double distance computation with normalized objective values (i.e. between 0 and 1).
 */
template < class MOEOT , class Type = double >
class moeoNormalizedDistance : public moeoDistance < MOEOT , Type >
{
public:

	/** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctr
     */
    moeoNormalizedDistance()
    {
        bounds.resize(ObjectiveVector::Traits::nObjectives());
        // initialize bounds in case someone does not want to use them
        for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            bounds[i] = eoRealInterval(0,1);
        }
    }

    /**
     * Returns a very small value that can be used to avoid extreme cases (where the min bound == the max bound)
     */
    static double tiny()
    {
    	return 1e-6;
    }


    /**
     * Sets the lower and the upper bounds for every objective using extremes values for solutions contained in the population _pop
     * @param _pop the population
     */
    virtual void setup(const eoPop < MOEOT > & _pop)
    {
        double min, max;
        for (unsigned i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            min = _pop[0].objectiveVector()[i];
            max = _pop[0].objectiveVector()[i];
            for (unsigned j=1; j<_pop.size(); j++)
            {
                min = std::min(min, _pop[j].objectiveVector()[i]);
                max = std::max(max, _pop[j].objectiveVector()[i]);
            }
            // setting of the bounds for the objective i
            setup(min, max, i);
        }
    }


	/**
     * Sets the lower bound (_min) and the upper bound (_max) for the objective _obj
     * @param _min lower bound
     * @param _max upper bound
     * @param _obj the objective index
     */
    virtual void setup(double _min, double _max, unsigned _obj)
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
    virtual void setup(eoRealInterval _realInterval, unsigned _obj)
    {
        bounds[_obj] = _realInterval;
    }


protected:

    /** the bounds for every objective (bounds[i] = bounds for the objective i) */
    std::vector < eoRealInterval > bounds;

};


/**
 * A class allowing to compute an euclidian distance between two solutions in the objective space with normalized objective values (i.e. between 0 and 1).
 * A distance value then lies between 0 and sqrt(nObjectives).
 */
template < class MOEOT >
class moeoEuclideanDistance : public moeoNormalizedDistance < MOEOT >
{
public:

	/** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    
    
    /**
     * Returns the euclidian distance between _moeo1 and _moeo2 in the objective space
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
	const double operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
		double result = 0.0;
		double tmp1, tmp2;
        for (unsigned i=0; i<ObjectiveVector::nObjectives(); i++)
        {
        	tmp1 = (_moeo1.objectiveVector()[i] - bounds[i].minimum()) / bounds[i].range();
        	tmp2 = (_moeo2.objectiveVector()[i] - bounds[i].minimum()) / bounds[i].range();
        	result += (tmp1-tmp2) * (tmp1-tmp2);
        }
        return sqrt(result);
    }


private:

    /** the bounds for every objective */
    using moeoNormalizedDistance < MOEOT > :: bounds;

};

/**
 * A class allowing to compute the Manhattan distance between two solutions in the objective space normalized objective values (i.e. between 0 and 1).
 * A distance value then lies between 0 and nObjectives.
 */
template < class MOEOT >
class moeoManhattanDistance : public moeoNormalizedDistance < MOEOT >
{
public:

	/** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    
    
    /**
     * Returns the Manhattan distance between _moeo1 and _moeo2 in the objective space
     * @param _moeo1 the first solution
     * @param _moeo2 the second solution
     */
	const double operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
    {
		double result = 0.0;
		double tmp1, tmp2;
        for (unsigned i=0; i<ObjectiveVector::nObjectives(); i++)
        {
        	tmp1 = (_moeo1.objectiveVector()[i] - bounds[i].minimum()) / bounds[i].range();
        	tmp2 = (_moeo2.objectiveVector()[i] - bounds[i].minimum()) / bounds[i].range();
        	result += fabs(tmp1-tmp2);
        }
        return result;
    }


private:

    /** the bounds for every objective */
    using moeoNormalizedDistance < MOEOT > :: bounds;

};

#endif /*MOEODISTANCE_H_*/
