// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoRandomSelect.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEORANDOMSELECT_H_
#define MOEORANDOMSELECT_H_

#include <moeoSelectOne.h>
#include <eoRandomSelect.h>

/**
 * Selection strategy that selects only one element randomly from a whole population. Neither the fitness nor the diversity of the individuals is required here.
 */
template < class MOEOT > class moeoRandomSelect:public moeoSelectOne < MOEOT >, public eoRandomSelect <MOEOT >
{
public:

	/**
	 * CTor.
	 */
	moeoRandomSelect(){}
	
  /*
   * Do nothing: we don't need to evaluate the fitness and the diversity; we only select one individual at random.
   */
  void setup (eoPop < MOEOT > &_pop)
  {
    // do nothing
  }

     /**
      * Return one individual at random.  // Need to have a "const" pop ?
      */
  const MOEOT & operator () (const eoPop < MOEOT > &_pop)
  {

    eoRandomSelect < MOEOT >::operator ()(_pop);
  }

};

#endif /*MOEORANDOMSELECT_H_ */
