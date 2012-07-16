/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoScalarFitness.h

 (c) Maarten Keijzer (mkeijzer@mad.scientist.com) and GeNeura Team, 1999, 2000

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
 */


#ifndef eoScalarFitness_h
#define eoScalarFitness_h

#include <functional>
#include <iostream>

/** @addtogroup Evaluation
 * @{
 */

/**
 * Wraps a scalar fitness values such as a double or int, with the option of
 * maximizing (using less<ScalarType>) or minimizing (using greater<ScalarType>)
 *
 * It overrides operator<() to use the Compare template argument. Thus, if you
 * need to compare if an indiv1 is "better" than an indiv2, you can use:
 *     if( indiv1 > indiv2 ) {
 *          // indiv1 is better
 *      } else {
 *          // indiv2 is better
 *      }
 *
 * Suitable constructors and assignments and casts are defined to work
 * with this quantity as if it were a ScalarType.
*/
template <class ScalarType, class Compare >
class eoScalarFitness
{
    public :

    eoScalarFitness() : value() {}
    eoScalarFitness(const eoScalarFitness& other) : value(other.value) {}
    eoScalarFitness(const ScalarType& v) : value(v) {}

    eoScalarFitness& operator=(const eoScalarFitness& other)
    { value = other.value; return *this; }
    eoScalarFitness& operator=(const ScalarType& v)
    { value = v; return *this; }

    /** Conversion operator: it permits to use a fitness instance as  its  scalar
     * type, if needed. For example, this is possible:
     *     eoScalarFitness<double,std::less<double> > fit;
     *     double val = 1.0;
     *     fit = val;
     *     val = fit;
     */
    operator ScalarType(void) const { return value; }

    /// Comparison, using less by default
    bool operator<(const eoScalarFitness& other) const
    { return Compare()(value, other.value); }

        /// Comparison, using less by default
        //  needed for MSVC 8 (MSVC 2005) added by J.Eggermont 20-11-2006
    bool operator<(const ScalarType& other) const
    { return Compare()(value, other); }

    // implementation of the other operators
    bool operator>( const eoScalarFitness<ScalarType, Compare>& y ) const  { return y < *this; }
    // implementation of the other operators
    bool operator<=( const eoScalarFitness<ScalarType, Compare>& y ) const { return !(*this > y); }
    // implementation of the other operators
    bool operator>=(const eoScalarFitness<ScalarType, Compare>& y ) const { return !(*this < y); }


    private :
        ScalarType value;
};
/** @example t-eofitness.cpp
 *
 */


/**
Typedefs for fitness comparison, Maximizing Fitness compares with less,
and minimizing fitness compares with greater. This because we want ordinary
fitness values (doubles) to be equivalent with Maximizing Fitness, and
comparing with less is the default behaviour.
*/
#if defined(__CUDACC__)
typedef eoScalarFitness<float, std::less<float> >    eoMaximizingFitness;
typedef eoScalarFitness<float, std::greater<float> > eoMinimizingFitness;
#else
typedef eoScalarFitness<double, std::less<double> >    eoMaximizingFitness;
typedef eoScalarFitness<double, std::greater<double> > eoMinimizingFitness;
#endif

template <class F, class Cmp>
std::ostream& operator<<(std::ostream& os, const eoScalarFitness<F, Cmp>& f)
{
    os << (F) f;
    return os;
}

template <class F, class Cmp>
std::istream& operator>>(std::istream& is, eoScalarFitness<F, Cmp>& f)
{
    F value;
    is >> value;
    f = value;
    return is;
}

/** @} */
#endif
