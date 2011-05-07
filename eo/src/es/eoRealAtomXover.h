/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealAtomXover.h : helper classes for std::vector<real> crossover
// (c) Marc Schoenauer 2001

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

    Contact: marc.schoenauer@polytechnique.fr http://eeaax.cmap.polytchnique.fr/
 */
//-----------------------------------------------------------------------------

/** Some basic atomic crossovers for doubles
 *
 * Are used in all ES specifici crossovers
 * and will be in more general stuff, using the generic crossovers
 */

#ifndef _eoRealAtomXover_H
#define _eoRealAtomXover_H

#include <utils/eoRNG.h>

#include <eoOp.h>

/**
  Discrete crossover == exchange of values
 *
 * @ingroup Real
 * @ingroup Variators
*/
class eoDoubleExchange: public eoBinOp<double>
{
public:
  /**
   * (Default) Constructor.
   */
  eoDoubleExchange() {}

  /// The class name. Used to display statistics
  virtual std::string className() const { return "eoDoubleExchange"; }

  /**
     Exchanges or not the values
   */
  bool operator()(double& r1, const double& r2)
  {
    if (eo::rng.flip())
      if (r1 != r2)        // if r1 == r2 you must return false
        {
          r1 = r2;
          return true;
        }
    return false;
  }

};

/**
  Intermediate crossover == linear combination
 *
 * @ingroup Real
 * @ingroup Variators
*/
class eoDoubleIntermediate: public eoBinOp<double>
{
public:
  /**
   * (Default) Constructor.
   */
  eoDoubleIntermediate() {}

  /// The class name. Used to display statistics
  virtual std::string className() const { return "eoDoubleIntermediate"; }

  /**
     Linear combination of both parents
   */
  bool operator()(double& r1, const double& r2)
  {
    double alpha = eo::rng.uniform();
    r1 = alpha * r2 + (1-alpha) * r1;
    return true;
  }

};

#endif
