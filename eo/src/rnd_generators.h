/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  rnd_generators.h
    Some utility functors for generating random generators:
        uniform_generator : generates uniform floats or doubles
        random_generator  : generates unsigneds, ints etc.
        normal_generator  : normally distributed floats or doubles

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

#ifndef eoRND_GENERATORS_H
#define eoRND_GENERATORS_H

#include "eoRNG.h"

/**
   The class uniform_generator can be used in the STL generate function
   to easily generate random floats and doubles between [0, _max). _max
   defaults to 1.0
*/
template <class T = double> class uniform_generator
{
  public :
    uniform_generator(T _max = T(1.0), eoRng& _rng = rng) : maxim(_max), uniform(_rng) {}
  
  virtual T operator()(void) { return (T) uniform.uniform(maxim); } 
  private :
    T maxim;
  eoRng& uniform;
};

/**
   The class random_generator can be used in the STL generate function
   to easily generate random ints between [0, _max).
*/
template <class T = uint32> class random_generator
{
  public :
    random_generator(int _max, eoRng& _rng = rng) : maxim(_max), random(_rng) {}
  
  virtual T operator()(void) { return (T) random.random(max); }
  
  private :
    T maxim;
  eoRng& random;
};

/**
   The class normal_generator can be used in the STL generate function
   to easily generate gaussian distributed floats and doubles. The user
   can supply a standard deviation which defaults to 1.
*/
template <class T = double> class normal_generator
{
  public :
    normal_generator(T _stdev = T(1.0), eoRng& _rng = rng) : stdev(_stdev), normal(_rng) {}
  
  virtual T operator()(void) { return (T) normal.normal(stdev); }
  
  private :
    T stdev;
  eoRng& normal;
};

#endif