/*
 * <moeoHypervolumeBinaryMetric.h>
 * Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
 * (C) OPAC Team, LIFL, 2002-2007
 *
 * July 2013: Bug fix in the recursive call of hypervolume (corrected thanks to Yann Semet and Dimo Brockhoff)
 *
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 * ParadisEO WebSite : http://paradiseo.gforge.inria.fr
 * Contact: paradiseo-help@lists.gforge.inria.fr
 *
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
 *
 * This indicator is based on the hypervolume concept introduced in
 * Zitzler, E., Thiele, L.: Multiobjective Optimization Using Evolutionary Algorithms - A Comparative Case Study.
 * Parallel Problem Solving from Nature (PPSN-V), pp.292-301 (1998).
 *
 * This code is adapted from the PISA implementation of IBEA (http://www.tik.ee.ethz.ch/sop/pisa/)
 *
 */
template < class ObjectiveVector >
class moeoHypervolumeBinaryMetric : public moeoNormalizedSolutionVsSolutionBinaryMetric < ObjectiveVector, double >
{
public:
    
    /**
     * Ctor
     * @param _rho value used to compute the reference point from the worst values for each objective (default : 1.1)
     */
    moeoHypervolumeBinaryMetric(double _rho = 1.1) : moeoNormalizedSolutionVsSolutionBinaryMetric<ObjectiveVector, double>(), rho(_rho)
    {
        // consistency check
        if (rho < 1)
        {
            eo::log << eo::warnings << "Warning, value used to compute the reference point rho for the hypervolume calculation must not be smaller than 1, adjusted to 1" << std::endl;
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
        // transform maximizing objectives into minimizing objectives
        ObjectiveVector o1 = _o1;
        ObjectiveVector o2 = _o2;
        for (unsigned int i=0; i<ObjectiveVector::Traits::nObjectives(); i++)
        {
            if (ObjectiveVector::Traits::maximizing(i))
            {
                o1[i] = bounds[i].maximum() - o1[i] + bounds[i].minimum();
                o2[i] = bounds[i].maximum() - o2[i] + bounds[i].minimum();
            }
        }
        // if _o2 is dominated by _o1
        if ( paretoComparator(_o2,_o1) )
        {
            result = - hypervolume(o1, o2, ObjectiveVector::Traits::nObjectives()-1);
        }
        else
        {
            result = hypervolume(o2, o1, ObjectiveVector::Traits::nObjectives()-1);
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
                result = hypervolume(_o1, _o2, _obj-1) * (max - v1) / range;
            }
        }
        return result;
    }
    
};

#endif /*MOEOHYPERVOLUMEBINARYMETRIC_H_*/
