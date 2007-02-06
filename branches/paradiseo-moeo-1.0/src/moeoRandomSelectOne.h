// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoRandomSelectOne.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEORANDOMSELECTONE_H_
#define MOEORANDOMSELECTONE_H_

#include <moeoSelectOne.h>
#include <eoRandomSelect.h>

/**
 * Selection strategy that selects only one element randomly from a whole population
 */
template < class MOEOT >
class moeoRandomSelectOne : public moeoSelectOne < MOEOT >, public eoRandomSelect < MOEOT > {};

#endif /*MOEORANDOMSELECTONE_H_*/
