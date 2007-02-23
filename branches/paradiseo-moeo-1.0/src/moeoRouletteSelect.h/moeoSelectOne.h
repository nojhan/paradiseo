// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoSelectOne.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOSELECTONE_H_
#define MOEOSELECTONE_H_

#include <eoSelectOne.h>

/**
 * Selection strategy for multi-objective optimization that selects only one element from a whole population
 */
template < class MOEOT > class moeoSelectOne : public eoSelectOne < MOEOT > {};

#endif /*MOEOSELECTONE_H_ */
