// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "FlowShopOpCrossoverQuad.h"

// (c) OPAC Team, LIFL, April 2006

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: Arnaud.Liefooghe@lifl.fr
*/

#ifndef _FlowShopOpCrossoverQuad_h
#define _FlowShopOpCrossoverQuad_h

#include <eoOp.h>

/**
 * Functor
 * Quadratic crossover operator for flow-shop (modify the both genotypes)
 */
class FlowShopOpCrossoverQuad:public eoQuadOp < FlowShop >
{

public:

  /**
   * default constructor
   */
  FlowShopOpCrossoverQuad ()
  {
  }

  /**
   * the class name (used to display statistics)
   */
  string className () const
  {
    return "FlowShopOpCrossoverQuad";
  }

  /**
   * eoQuad crossover - _genotype1 and _genotype2 are the (future) offspring, i.e. _copies_ of the parents
   * @param FlowShop & _genotype1  the first parent
   * @param FlowShop & _genotype2  the second parent
   */
  bool operator  () (FlowShop & _genotype1, FlowShop & _genotype2)
  {
    bool oneAtLeastIsModified;

    // parents
    vector < unsigned >parent1 = _genotype1.getScheduling ();
    vector < unsigned >parent2 = _genotype2.getScheduling ();

    // computation of the 2 random points
    unsigned point1, point2;
    do
      {
	point1 = rng.random (min (parent1.size (), parent2.size ()));
	point2 = rng.random (min (parent1.size (), parent2.size ()));
      }
    while (fabs ((double) point1 - point2) <= 1);

    // computation of the offspring
    vector < unsigned >offspring1 =
      generateOffspring (parent1, parent2, point1, point2);
    vector < unsigned >offspring2 =
      generateOffspring (parent2, parent1, point1, point2);

    // does at least one genotype has been modified ?
    if ((parent1 != offspring1) || (parent2 != offspring2))
      {
	// update
	_genotype1.setScheduling (offspring1);
	_genotype2.setScheduling (offspring2);
	// at least one genotype has been modified
	oneAtLeastIsModified = true;
      }
    else
      {
	// no genotype has been modified
	oneAtLeastIsModified = false;
      }

    // return 'true' if at least one genotype has been modified
    return oneAtLeastIsModified;
  }


private:

  /**
   * generation of an offspring by a 2 points crossover
   * @param vector<unsigned> _parent1  the first parent
   * @param vector<unsigned> _parent2  the second parent
   * @param unsigned_point1  the first point
   * @param unsigned_point2  the second point
   */
  vector < unsigned >generateOffspring (vector < unsigned >_parent1,
					vector < unsigned >_parent2,
					unsigned _point1, unsigned _point2)
  {
    vector < unsigned >result = _parent1;
    vector < bool > taken_values (result.size (), false);
    if (_point1 > _point2)
      swap (_point1, _point2);

    /* first parent */
    for (unsigned i = 0; i <= _point1; i++)
      {
	// result[i] == _parent1[i]
	taken_values[_parent1[i]] = true;
      }
    for (unsigned i = _point2; i < result.size (); i++)
      {
	// result[i] == _parent1[i]
	taken_values[_parent1[i]] = true;
      }

    /* second parent */
    unsigned i = _point1 + 1;
    unsigned j = 0;
    while (i < _point2 && j < _parent2.size ())
      {
	if (!taken_values[_parent2[j]])
	  {
	    result[i] = _parent2[j];
	    i++;
	  }
	j++;
      }

    return result;
  }

};

#endif
