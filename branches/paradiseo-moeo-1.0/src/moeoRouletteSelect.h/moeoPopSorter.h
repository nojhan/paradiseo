// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoPopSorter.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------


#ifndef MOEOPOPSORTER_H_
#define MOEOPOPSORTER_H_

#include <moeoComparator.h>

/**
 * Sorter.
 */
template < class MOEOT > class moeoPopSorter
{

public:

	/** Ctor taking a moeoComparator */
moeoPopSorter (moeoComparator _comparator,):comparator (_comparator)
  {
  }


	/**
	 * Sort a population by applying the comparator
	 */
  const bool operator () (eoPop < MOEOT > &_pop)
  {
    // eval fitness and diversity : need "assignement" classes

    // apply comparator
    std::sort (_pop.begin (), _pop.end (), comparator);
  }

protected:
  /** Comparator attribute */
  moeoComparator & comparator;

};

#endif /*MOEOPOPSORTER_H_ */
