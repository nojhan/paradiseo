// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpMutationExchange.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

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
  bool operator   () (FlowShop & _genotype)
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
