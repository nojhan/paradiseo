/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoUniformInit.h

 (c) Maarten Keijzer, GeNeura Team, Marc Schoenauer, 1999 - 2002

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
             Marc.Schoenauer@inria.fr
 */

//-----------------------------------------------------------------------------
/** Copied from the eoRndGenerators to have objects deriving from eoInit
 * As the whole initialization mechanism in EO is based on eoInit rather than
 * eoRndGenerators, we might as well have these directly written without
 * overhead

@class eoUniformInit     uniform initialization for doubles, floats, ints, ...
@class eoBooleanInit     biased init for booleans
@class eoNormalInit      normal intialization for doubles and floats
@class eoNegExpInit      negative exponential distributions    "
*/

#ifndef eoUniformInit_h
#define eoUniformInit_h

#include "eoRNG.h"
#include "eoInit.h"
#include "eoRealBounds.h"
#include <stdexcept>


/** @addtogroup Initializators
 * @{
 */

using eo::rng;

/**
   The class eoUniformInit can be used in the STL apply function
   to easily randomize floats and doubles. It can also
   be used for ints and unsigneds by virtue of the static_cast

   Also present is a specialization for boolean, that will
   ignore the minima and maxima that are possibly set and will
   return an unbiased flip of a coin. For a biased flip, use the eoBoolean

   either in [0, _max) if only 1 value (_max) is given
                       (or none, as _max defaults to 1.0)
   or in [_min,_max) if 2 values are given (_min, _max)

*/
template <class T = double> class eoUniformInit : public eoInit<T>
{
  public :
  /** Ctor with only a max bound */
  eoUniformInit(T _max = T(1.0), eoRng& _rng = rng) :
    minim(T(0.0)), range(_max), uniform(_rng)
  {}

  /** Ctor with an eoRealBound */
  eoUniformInit(eoRealBounds & _bound, eoRng& _rng = rng) :
    minim(_bound.minimum()), range(_bound.range()), uniform(_rng)
  {}

  /** Ctor with explicit min and max */
  eoUniformInit(T _min, T _max, eoRng& _rng = rng) :
    minim(_min), range(_max-_min), uniform(_rng)
  {
    if (_min>_max)
      throw std::logic_error("Min is greater than Max in uniform_generator");
  }

  /**
     Generates the number, uses a static_cast to get the right behaviour
     for ints and unsigneds
  */
  void operator()(T & _t)
  {
    _t = minim+static_cast<T>(uniform.uniform(range));
  }

private :
  T minim;
  T range;
  eoRng& uniform;
};


/** Specialization for bool, does an unbiased coin flip */
template <>
inline void eoUniformInit<bool>::operator()(bool & _b)
{
    _b = uniform.flip(0.5);
}

/**
   The class eoBooleanInit can be used in the STL apply function
   to easily generate random booleans with a specified bias
*/
class eoBooleanInit : public eoInit<bool>
{
  public :
  eoBooleanInit(float _bias = 0.5, eoRng& _rng = rng) : bias(_bias), gen(_rng) {}

  void operator()(bool & _b) { _b = gen.flip(bias); }
  private :
  float bias;
  eoRng& gen;
};

/**
   The class normal_generator can be used in the STL generate function
   to easily generate gaussian distributed floats and doubles. The user
   can supply a standard deviation which defaults to 1.
*/
template <class T = double> class eoNormalInit : public eoInit<T>
{
  public :
    eoNormalInit(T _stdev = T(1.0), eoRng& _rng = rng) : stdev(_stdev), normal(_rng) {}

  void operator()(T & _t) { _t = (T) normal.normal(stdev); }

  private :
    T stdev;
  eoRng& normal;
};

/**
   The class negexp_generator can be used in the STL generate function
   to easily generate negative exponential distributed floats and doubles. The user
   can supply a mean.
*/
template <class T = double> class eoNegExpInit : public eoInit<T>
{
  public :
    eoNegExpInit(T _mean = 1.0, eoRng& _rng = rng) : mean(_mean), negexp(_rng) {}

  void operator()(T & _t) { _t = (T) negexp.negexp(mean); }

  private :
    T mean;
  eoRng& negexp;
};

/** @} */
#endif
