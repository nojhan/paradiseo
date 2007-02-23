// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoDiversityAssignment.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEODIVERSITYASSIGNMENT_H_
#define MOEODIVERSITYASSIGNMENT_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor that sets the diversity values of a whole population
 */
template < class MOEOT > class moeoDiversityAssignment:public eoUF < eoPop < MOEOT > &,
  void >
{
};


/**
 * moeoDummyDiversityAssignment is a moeoDiversityAssignment which gives 0 as the diversity for the whole population.
 */
template < class MOEOT > class moeoDummyDiversityAssignment:public moeoDiversityAssignment <
  MOEOT >
{
  // main operator
  void operator () (eoPop < MOEOT > &_pop)
  {
    for (int idx = 0; idx < _pop.size (); idx++)
      {
	// set the diversity to 0
	_pop[idx].diversity (0);
      }
  }

};



#endif /*MOEODIVERSITYASSIGNMENT_H_ */
