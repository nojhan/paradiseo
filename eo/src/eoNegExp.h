// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoNegExp.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EONEGEXP_H
#define _EONEGEXP_H

//-----------------------------------------------------------------------------

#include <time.h>
#include <stdlib.h> 

#include <math.h>

#include <eoRnd.h> // for base class

//-----------------------------------------------------------------------------
// Class eoNegExp
//-----------------------------------------------------------------------------

/// Generates random numbers using a negative exponential distribution
template<class T>
class eoNegExp: public eoRnd<T>
{
 public:
  /**
   * Default constructor.
   * @param _mean  Dsitribution mean
   */
  eoNegExp(T _mean): eoRnd<T>(), mean(_mean) {};

  /**
   * Copy constructor.
   * @param _rnd the copyee
   */
  eoNegExp( const eoNegExp& _rnd): eoRnd<T>( _rnd), mean(_rnd.mean) {};
  
  /// Returns an uniform dandom number over the interval [min, max).
  virtual T operator()() { return - mean*log((double)rand() / RAND_MAX); }
  
 private:
  T mean;
};

//-----------------------------------------------------------------------------

#endif
