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

/**
 * eoScalarFitness<ScalarType, Compare = less<ScalarType> >:
 * Wraps a scalar fitness values such as a double or int, with the option of 
 * maximizing (using less<ScalarType>) or miniziming (using greater<ScalarType>)
 * this quantity in EO. 
 *
 * It overrides operator<() to use the Compare template argument
 *
 * Suitable constructors and assignments and casts are defined to work 
 * with this quantity as if it were a ScalarType.
*/

template <class ScalarType, class Compare = less<ScalarType> > 
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

    operator ScalarType(void) const { return value; }

    /// Comparison, using less by default
    bool operator<(const eoScalarFitness& other) const 
    { return Compare()(value, other.value); }

    private :
        ScalarType value;
};

template <class F, class Cmp>
std::ostream& operator<<(std::ostream& os, const eoScalarFitness<F, Cmp>& f)
{
    os << (F) f;
    return os;
}

template <class F, class Cmp>
std::istream& operator>>(istream& is, eoScalarFitness<F, Cmp>& f)
{
    F value;
    is >> value;
    f = value;
    return is;
}

#endif
