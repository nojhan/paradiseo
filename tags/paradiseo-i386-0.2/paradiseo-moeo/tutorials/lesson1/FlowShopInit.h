// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopInit.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _FlowShopInit_h
#define _FlowShopInit_h

#include <eoInit.h>


/**
 * Functor
 * Initialisation of a random genotype built by the default constructor of the eoFlowShop class
 */
class FlowShopInit:public eoInit < FlowShop >
{

public:

  /** 
   * constructor
   * @param const unsigned _N  the number of jobs to schedule
   */
  FlowShopInit (const unsigned _N)
  {
    N = _N;
  }

  /**
   * randomize a genotype
   * @param FlowShop & _genotype  a genotype that has been default-constructed
   */
  void operator   () (FlowShop & _genotype)
  {
    // scheduling vector
    vector < unsigned >scheduling (N);
    // initialisation of possible values
    vector < unsigned >possibles (N);
    for (unsigned i = 0; i < N; i++)
      possibles[i] = i;
    // random initialization
    unsigned rInd;		// random index
    for (unsigned i = 0; i < N; i++)
      {
	rInd = (unsigned) rng.uniform (N - i);
	scheduling[i] = possibles[rInd];
	possibles[rInd] = possibles[N - i - 1];
      }
    _genotype.setScheduling (scheduling);
    _genotype.invalidate ();	// IMPORTANT in case the _genotype is old
  }


private:
  /** the number of jobs (size of a scheduling vector) */
  unsigned N;

};

#endif
