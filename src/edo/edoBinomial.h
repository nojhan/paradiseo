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

#ifndef _edoBinomial_h
#define _edoBinomial_h

#include <vector>

#include "edoDistrib.h"

/** @defgroup Binomial Binomial
 * A binomial distribution that model marginal probabilities across boolean
 * variables.
 *
 * @ingroup Distributions
 */

/** A binomial distribution that model marginal probabilities across variables.
 *
 * @ingroup Distributions
 * @ingroup Binomial
 */
template<class EOT, class T=std::vector<double> >
class edoBinomial : public edoDistrib<EOT>, public T
{
public:
    typedef double AtomType; // FIXME use container atom type

    /** This constructor takes an initial vector of probabilities.
     *  Use it if you have prior knowledge.
     */
    edoBinomial( T initial_probas ) : T(initial_probas) {}

    /** This constructor makes no assumption about initial probabilities.
     *  Every probabilities are set to 0.0.
     */
    edoBinomial( size_t dim, double p = 0.0 ) : T(dim,p) {}

    /** Constructor without any assumption.
     * Will create a vector of size 1 with a probability of 0.0.
     */
    edoBinomial() : T(1,0.0) {}
};

#endif // !_edoBinomial_h

