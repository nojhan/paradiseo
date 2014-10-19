// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoDistance.h
// (c) GeNeura Team, 1998, Marc Schoenauer 2001
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
             Marc.Schoenauer@polytechnique.fr
 */
//-----------------------------------------------------------------------------

#ifndef _eoDistance_H
#define _eoDistance_H

#include <eoFunctor.h>

/** @addtogroup Stats
 * @{
 * */

/**
    This is a generic class for distance functors:
    takes 2 things and returns a double
*/
template< class EOT >
class eoDistance : public eoBF<const EOT &, const EOT &, double>
{};


/**
    This is a generic class for Euclidain distance (L2 norm) computation:
    assumes the 2 things are std::vectors of something that is double-castable
*/
template< class EOT >
class eoQuadDistance : public eoDistance<EOT>
{
public:
  double operator()(const EOT & _v1, const EOT & _v2)
  {
     double sum=0.0;
    for (unsigned i=0; i<_v1.size(); i++)
      {
        double r = static_cast<double> (_v1[i]) - static_cast<double> (_v2[i]);
        sum += r*r;
      }
    return sqrt(sum);
  }
};

/**
    This is a generic class for L1 distance computation:
    assumes the 2 things are std::vectors of something
    that is double-castable
    For bitstrings, this is the Hamming distance
*/
template< class EOT >
class eoHammingDistance : public eoDistance<EOT>
{
public:
  double operator()(const EOT & _v1, const EOT & _v2)
  {
     double sum=0.0;
    for (unsigned i=0; i<_v1.size(); i++)
      {
        double r = static_cast<double> (_v1[i]) - static_cast<double> (_v2[i]);
        sum += fabs(r);
      }
    return sum;
  }
};

/* this distance measures the difference in fitness
 * I am not sure it can be of any use, though ...
 * except for some testing
 */
template< class EOT >
class eoFitnessDistance : public eoDistance<EOT>
{
public:
  double operator()(const EOT & _v1, const EOT & _v2)
  {
    double d = _v1.fitness() - _v2.fitness();
    return sqrt(d*d);
  }
};


/** @} */

#endif
