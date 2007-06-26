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
 * Selection strategy that selects only one element randomly from a whole population.
 */
template < class MOEOT > class moeoRandomSelect:public moeoSelectOne < MOEOT >, public eoRandomSelect <MOEOT >
{
public:

    /**
     * Ctor.
     */
    moeoRandomSelect(){}


    /**
     * Return one individual at random by using an eoRandomSelect.
     */
    const MOEOT & operator () (const eoPop < MOEOT > &_pop)
    {
        return eoRandomSelect < MOEOT >::operator ()(_pop);
    }

};

#endif /*MOEORANDOMSELECT_H_ */
