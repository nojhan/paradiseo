// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpMutationShift.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOPMUTATIONSHIFT_H_
#define FLOWSHOPOPMUTATIONSHIFT_H_

#include <eoOp.h>
#include "FlowShop.h"

/**
 * Functor
 * Shift mutation operator for flow-shop
 */
class FlowShopOpMutationShift: public eoMonOp<FlowShop> {

public:
  
  /**
   * default constructor
   */
  FlowShopOpMutationShift() {}
  
  /**
   * the class name (used to display statistics)
   */
  string className() const { 
    return "FlowShopOpMutationShift";
  }

  /**
   * modifies the parent with a shift mutation
   * @param FlowShop & _genotype  the parent genotype (will be modified)
   */
  bool operator()(FlowShop & _genotype) {
    bool isModified;
    int direction;
    unsigned tmp;
    
    // schedulings
    vector<unsigned> initScheduling   = _genotype.getScheduling();
    vector<unsigned> resultScheduling = initScheduling;
    
    // computation of the 2 random points
    unsigned point1, point2;
    do {
      point1 = rng.random(resultScheduling.size());
      point2 = rng.random(resultScheduling.size());
    } while (point1 == point2);
    
    // direction
    if (point1 < point2) direction = 1;
    else direction = -1;
    // mutation
    tmp = resultScheduling[point1];
    for(unsigned i=point1 ; i!=point2 ; i+=direction)
      resultScheduling[i] = resultScheduling[i+direction];
    resultScheduling[point2] = tmp;
    
    // update (if necessary)
    if (resultScheduling != initScheduling) {
      // update
      _genotype.setScheduling(resultScheduling);
      // the genotype has been modified
      isModified = true;
    }
    else {
      // the genotype has not been modified
      isModified = false;     
    }

    // return 'true' if the genotype has been modified
    return isModified;
  }

};

#endif /*FLOWSHOPOPMUTATIONSHIFT_H_*/
