// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// EOUniform.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EOUNIFORM_H
#define _EOUNIFORM_H

//-----------------------------------------------------------------------------

#include <time.h>
#include <stdlib.h> 

#include <eoRnd.h>
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
    : eoRnd<T>(), minim(_min), diff(_max - _min) {}

  /**
   * copy constructor.
   * @param _rnd the other rnd
   */
  eoUniform( const eoUniform& _rnd)
    : eoRnd<T>( _rnd), minim(_rnd.minim), diff(_rnd.diff) {}
  
  /// Returns an uniform dandom number over the interval [min, max).
  virtual T operator()() { 
	  return minim+ T( (diff * rand() )/ RAND_MAX); 
  }
  
 private:
  T minim;
  double diff;
};

//-----------------------------------------------------------------------------

#endif
