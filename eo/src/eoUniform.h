/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

  -----------------------------------------------------------------------------
  eoUniform.h
    Uniform random number generator; 
  (c) GeNeura Team, 1998
 
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

#ifndef _EOUNIFORM_H
#define _EOUNIFORM_H

//-----------------------------------------------------------------------------

#include <eoRnd.h>
#include <utils/eoRNG.h>

//-----------------------------------------------------------------------------
// Class eoUniform
//-----------------------------------------------------------------------------

/// Generates uniform random number over the interval [min, max)
template<class T>
class eoUniform: public eoRnd<T>
{
 public:
  /**
   * Default constructor.
   * @param _min  The minimum value in the interval.
   * @param _max  The maximum value in the interval.
   */
  eoUniform(T _min = 0, T _max = 1)
    : eoRnd<T>(), min(_min), diff(_max - _min) {}

  /**
   * copy constructor.
   * @param _rnd the other rnd
   */
  eoUniform( const eoUniform& _rnd)
    : eoRnd<T>( _rnd), min(_rnd.minim), diff(_rnd.diff) {}
  
  /** Returns an uniform random number over the interval [min, max)
      Uses global rng object */
  virtual T operator()() { 
    return min + T( rng.uniform( diff ) );  
  }
  
 private:
  T min;
  double diff;
};

template<>
class eoUniform<bool>: public eoRnd<bool>
{
 public:
  /**
   * Default constructor.
   * @param _min  The minimum value in the interval.
   * @param _max  The maximum value in the interval.
   */
  eoUniform(bool _min = false, bool _max = true)
    : eoRnd<bool>() {}

  /** Returns an uniform random number over the interval [min, max)
      Uses global rng object */
  virtual bool operator()() { 
    return rng.flip(0.5);  
  }
  
 private:
  T min;
  double diff;
};

//-----------------------------------------------------------------------------

#endif
