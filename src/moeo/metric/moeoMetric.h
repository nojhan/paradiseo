/*
* <moeoMetric.h>
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

#ifndef MOEOMETRIC_H_
#define MOEOMETRIC_H_

#include <vector>
#include "../../eo/eoFunctor.h"

/**
 * Base class for performance metrics (also known as quality indicators).
 */
class moeoMetric : public eoFunctorBase
  {};


/**
 * Base class for unary metrics.
 */
template < class A, class R >
class moeoUnaryMetric : public eoUF < A, R >, public moeoMetric
  {};


/**
 * Base class for binary metrics.
 */
template < class A1, class A2, class R >
class moeoBinaryMetric : public eoBF < A1, A2, R >, public moeoMetric
  {};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a single solution's objective vector.
 */
template < class ObjectiveVector, class R >
class moeoSolutionUnaryMetric : public moeoUnaryMetric < const ObjectiveVector &, R >
  {};


/**
 * Base class for unary metrics dedicated to the performance evaluation of a Pareto set (a vector of objective vectors)
 */
template < class ObjectiveVector, class R >
class moeoVectorUnaryMetric : public moeoUnaryMetric < const std::vector < ObjectiveVector > &, R >
  {};


/**
 * Base class for binary metrics dedicated to the performance comparison between two solutions's objective vectors.
 */
template < class ObjectiveVector, class R >
class moeoSolutionVsSolutionBinaryMetric : public moeoBinaryMetric < const ObjectiveVector &, const ObjectiveVector &, R >
  {};


/**
 * Base class for binary metrics dedicated to the performance comparison between two Pareto sets (two vectors of objective vectors)
 */
template < class ObjectiveVector, class R >
class moeoVectorVsVectorBinaryMetric : public moeoBinaryMetric < const std::vector < ObjectiveVector > &, const std::vector < ObjectiveVector > &, R >
  {};


#endif /*MOEOMETRIC_H_*/
