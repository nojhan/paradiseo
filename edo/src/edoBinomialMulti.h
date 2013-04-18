/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2013 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _edoBinomialMulti_h
#define _edoBinomialMulti_h

#include "edoBinomial.h"

#ifdef WITH_EIGEN // FIXME: provide an uBLAS implementation
#include <Eigen/Dense>

/** A 2D binomial distribution modeled as a matrix.
 *
 * i.e. a container of binomial distribution.
 *
 * @ingroup Distributions
 * @ingroup Binomial
 */
template<class EOT, class T=Eigen::MatrixXd>
class edoBinomialMulti : public edoDistrib<EOT>, public T
{
public:
    /** This constructor takes an initial matrix of probabilities.
     *  Use it if you have prior knowledge.
     */
    edoBinomialMulti( T initial_probas )
        : T(initial_probas) {}

    /** Initialize all the probabilities to a constant
     *
     * 0.5 by default
     */
    edoBinomialMulti( unsigned int rows, unsigned int cols, double proba=0.5 )
        : T::Constant(rows,cols,proba) {}

    /** Constructor without any assumption.
     */
    edoBinomialMulti() {}
};

#endif // WITH_EIGEN
#endif // !_edoBinomialMulti_h

