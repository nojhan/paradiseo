// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoGenerationalReplacement.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOGENERATIONALREPLACEMENT_H_
#define MOEOGENERATIONALREPLACEMENT_H_

#include <eoReplacement.h>
#include <moeoReplacement.h>

/**
 * Generational replacement: only the new individuals are preserved.
 */
template < class MOEOT >
class moeoGenerationalReplacement : public moeoReplacement < MOEOT >, public eoGenerationalReplacement < MOEOT >
{
public:
	
	/**
	 * Swaps _parents and _offspring
	 * @param _parents the parents population
	 * @param _offspring the offspring population
	 */
	void operator()(eoPop < MOEOT > & _parents, eoPop < MOEOT > & _offspring)
	{
		eoGenerationalReplacement < MOEOT >::operator ()(_parents, _offspring);
	}

};

#endif /*MOEOGENERATIONALREPLACEMENT_H_*/
