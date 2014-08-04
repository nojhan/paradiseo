/*
* <moeoAdditiveEpsilonBinaryMetric.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Arnaud Liefooghe
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

#ifndef MOEOADDITIVEEPSILONBINARYMETRIC_H_
#define MOEOADDITIVEEPSILONBINARYMETRIC_H_

#include "moeoNormalizedSolutionVsSolutionBinaryMetric.h"

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
      for (unsigned int i=1; i<ObjectiveVector::Traits::nObjectives(); i++)
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
    double epsilon(const ObjectiveVector & _o1, const ObjectiveVector & _o2, const unsigned int _obj)
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
