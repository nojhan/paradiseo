// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "FlowShopOpMutationExchange.h"

// (c) OPAC Team, LIFL, March 2006

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

#ifndef _FlowShopOpMutationExchange_h
#define _FlowShopOpMutationExchange_h

#include <eoOp.h>


/**
 * Functor
 * Exchange mutation operator for flow-shop
 */
class FlowShopOpMutationExchange:public eoMonOp < FlowShop >
{

public:

  /**
   * default constructor
   */
  FlowShopOpMutationExchange ()
  {
  }

  /**
   * the class name (used to display statistics)
   */
  string className () const
  {
    return "FlowShopOpMutationExchange";
  }

  /**
   * modifies the parent with an exchange mutation
   * @param FlowShop & _genotype  the parent genotype (will be modified)
   */
  bool operator  () (FlowShop & _genotype)
  {
    bool isModified;

    // schedulings
    vector < unsigned >initScheduling = _genotype.getScheduling ();
    vector < unsigned >resultScheduling = _genotype.getScheduling ();

    // computation of the 2 random points
    unsigned point1, point2;
    do
      {
	point1 = rng.random (resultScheduling.size ());
	point2 = rng.random (resultScheduling.size ());
      }
    while (point1 == point2);

    // swap
    swap (resultScheduling[point1], resultScheduling[point2]);

    // update (if necessary)
    if (resultScheduling != initScheduling)
      {
	// update
	_genotype.setScheduling (resultScheduling);
	// the genotype has been modified
	isModified = true;
      }
    else
      {
	// the genotype has not been modified
	isModified = false;
      }

    // return 'true' if the genotype has been modified
    return isModified;
  }

};

#endif
