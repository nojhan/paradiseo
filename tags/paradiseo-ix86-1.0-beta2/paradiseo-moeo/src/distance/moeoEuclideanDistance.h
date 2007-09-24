// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEuclideanDistance.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOEUCLIDEANDISTANCE_H_
#define MOEOEUCLIDEANDISTANCE_H_

#include <math.h>
#include <distance/moeoNormalizedDistance.h>

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
        for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
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

#endif /*MOEOEUCLIDEANDISTANCE_H_*/
