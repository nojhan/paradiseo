// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoNormal.h
// (c) GeNeura Team, 1998
//-----------------------------------------------------------------------------

#ifndef _EONORMAL_H
#define _EONORMAL_H

//-----------------------------------------------------------------------------

#include <time.h>
#include <stdlib.h> 

#include <math.h>

#include <eoRnd.h> // for base class

//-----------------------------------------------------------------------------
// Class eoNormal
//-----------------------------------------------------------------------------

/// Generates random number using a normal distribution
template<class T>
class eoNormal: public eoRnd<T>
{
 public:
  /**
   * Default constructor.
   * @param _mean  Dsitribution mean
   * @param _sd    Standard Deviation
   */
  eoNormal(T _mean, T _sd)
    : eoRnd<T>(), mean(_mean), sd(_sd), phase(false) {}

  /**
   * Copy constructor.
   * @param _rnd the other one
   */
  eoNormal( const eoNormal& _rnd )
    : eoRnd<T>( _rnd), mean(_rnd.mean), sd(_rnd.sd), phase(false) {}
  
  /** Returns an uniform random number over the interval [min, max).
      @return an uniform random number over the interval [min, max).
   */
  virtual T operator()() {
    if (phase) { // Already have one stored up.
      phase = false;
      return T ( (sqRatio * q * sd) + mean );
    }

    double p, v;
    do {
      p = ((double)rand() / RAND_MAX)*2-1;
      q = ((double)rand() / RAND_MAX)*2-1;
      v = p*p + q*q;
    } while(v > 1.0 || v <0.25);

    sqRatio = sqrt(-2*log((double)rand() / RAND_MAX) / v);
    phase = true;
    return T( (sqRatio * p * sd) + mean );
  };

 private:
  T mean;
  T sd;
  bool phase;
  double sqRatio, q;
};

//-----------------------------------------------------------------------------

#endif
