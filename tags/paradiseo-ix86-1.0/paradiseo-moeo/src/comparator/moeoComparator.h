// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCOMPARATOR_H_
#define MOEOCOMPARATOR_H_

#include <eoFunctor.h>

/**
 * Functor allowing to compare two solutions.
 */
template < class MOEOT >
class moeoComparator : public eoBF < const MOEOT &, const MOEOT &, const bool > {};

#endif /*MOEOCOMPARATOR_H_*/
