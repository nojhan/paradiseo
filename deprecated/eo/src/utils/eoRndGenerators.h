/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoRndGenerators.h

  mimics the rnd_generator.h by putting the generators in the EO-hierarchy

 (c) Maarten Keijzer (mak@dhi.dk) and GeNeura Team, 1999, 2000

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

//-----------------------------------------------------------------------------

#ifndef eoRndGenerators_h
#define eoRndGenerators_h

#include "eoRNG.h"
#include <eoFunctor.h>
#include <stdexcept>

/** @defgroup Random Random number generation
 *
 * @ingroup Utilities
 * @{
 */

using eo::rng;

/**
  By popular demand re-introducing a base class for a family of
  random number generators. Derived members of this class are useful
  to initialize fixed/variable length genotypes that have an 'atomic' type
  in an indepent way (thus without employing any knowledge about the problem domain).

  See derived classes eoUniformGenerator, eoBooleanGenerator, eoNormalGenerator and eoNegExpGenerator
*/
template <class T>
class eoRndGenerator : public eoF<T>
{
  typedef T AtomType;
};

/**
   The class eoUniform can be used in the STL generate function
   to easily generate random floats and doubles. It can also
   be used for ints and unsigneds by virtue of the static_cast
   in the generator function.

   Also present is a specialization for boolean, that will
   ignore the minima and maxima that are possibly set and will
   return an unbiased flip of a coin. For a biased flip, use the eoBoolean

   either in [0, _max) if only 1 value (_max) is given
                       (or none, as _max defaults to 1.0)
   or in [_min,_max) if 2 values are given (_min, _max)
*/
template <class T = double> class eoUniformGenerator : public eoRndGenerator<T>
{
  // added new ctor with 2 params, and modified the data to minim and range
  // (was maxim only). MS 3/11/2000
  public :
    eoUniformGenerator(T _max = T(1.0), eoRng& _rng = rng) :
      minim(T(0.0)), range(_max), uniform(_rng) {}
    eoUniformGenerator(T _min, T _max, eoRng& _rng = rng) :
      minim(_min), range(_max-_min), uniform(_rng)
  {
    if (_min>_max)
      throw std::logic_error("Min is greater than Max in uniform_generator");
  }

  /**
  Generates the number, uses a static_cast to get the right behaviour
  for ints and unsigneds
  */
  T operator()(void) { return minim+static_cast<T>(uniform.uniform(range)); }
  /** @example t-eoUniform.cpp
   */

private :
  T minim;
  T range;
  eoRng& uniform;
};
/** @example t-eoRandom.cpp
 */


/// Specialization for bool, does an unbiased coin flip
template <>
inline bool eoUniformGenerator<bool>::operator()(void)
{
    return uniform.flip(0.5);
}

/**
   The class eoBooleanGenerator can be used in the STL generate function
   to easily generate random booleans with a specified bias
   \ingroup bitstring
*/
class eoBooleanGenerator : public eoRndGenerator<bool>
{
  public :
  eoBooleanGenerator(float _bias = 0.5, eoRng& _rng = rng) : bias(_bias), gen(_rng) {}

  bool operator()(void) { return gen.flip(bias); }
  private :
  float bias;
  eoRng& gen;
};

/**
   The class normal_generator can be used in the STL generate function
   to easily generate gaussian distributed floats and doubles. The user
   can supply a standard deviation which defaults to 1.
*/
template <class T = double> class eoNormalGenerator : public eoRndGenerator<T>
{
  public :
    eoNormalGenerator(T _stdev = T(1.0), eoRng& _rng = rng) : stdev(_stdev), normal(_rng) {}

  T operator()(void) { return (T) normal.normal(stdev); }

  private :
    T stdev;
  eoRng& normal;
};

/**
   The class negexp_generator can be used in the STL generate function
   to easily generate negative exponential distributed floats and doubles. The user
   can supply a mean.
*/
template <class T = double> class eoNegExpGenerator : public eoRndGenerator<T>
{
  public :
    eoNegExpGenerator(T _mean = 1.0, eoRng& _rng = rng) : mean(_mean), negexp(_rng) {}

  T operator()(void) { return (T) negexp.negexp(mean); }

  private :
    T mean;
  eoRng& negexp;
};

/** @} */
#endif
