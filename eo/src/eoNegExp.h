/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
 eoNegExp.h
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

#ifndef _EONEGEXP_H
#define _EONEGEXP_H

//-----------------------------------------------------------------------------

#include <math.h>

#include <eoRnd.h> // for base class
#include <eoRNG.h> // for base class

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
   * @param _mean  Distribution mean
   */
  eoNegExp(T _mean): eoRnd<T>(), mean(_mean) {};

  /**
   * Copy constructor.
   * @param _rnd the copyee
   */
  eoNegExp( const eoNegExp& _rnd): eoRnd<T>( _rnd), mean(_rnd.mean) {};
  
  /// Returns an uniform dandom number over the interval [min, max).
  virtual T operator()() { 
    return T( -mean*log((double)rng.rand() / rng.rand_max())); }
  
 private:
  T mean;
};

//-----------------------------------------------------------------------------

#endif

