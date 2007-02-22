// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOLS_H_
#define MOEOLS_H_

#include <eoFunctor.h>
#include <moeoArchive.h>

/**
 * Abstract class for local searches applied to multi-objective optimization.
 * Starting from only one solution, it produces a set of new non-dominated solutions.
 */
template < class MOEOT > class moeoLS:public eoBF < const MOEOT &, moeoArchive < MOEOT > &,
  void >
{
};

#endif /*MOEOLS_H_ */
