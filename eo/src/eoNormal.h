// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoNormal.h
// (c) GeNeura Team, 1998
/* 
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

#ifndef _EONORMAL_H
#define _EONORMAL_H

//-----------------------------------------------------------------------------

#include <math.h>
#include <eoRnd.h> // for base class
#include <eoRNG.h> // for random number generator

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
      p = ((double)rng.rand() / rng.rand_max())*2-1;
      q = ((double)rng.rand() / rng.rand_max())*2-1;
      v = p*p + q*q;
    } while(v > 1.0 || v <0.25);

    sqRatio = sqrt(-2*log((double)rand() / rng.rand_max()) / v);
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
