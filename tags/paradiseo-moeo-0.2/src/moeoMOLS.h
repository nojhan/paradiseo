// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoMOLS.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOMOLS_H_
#define MOEOMOLS_H_

#include <eoFunctor.h>
#include <moeoArchive.h>

/**
 * Abstract class for local searches applied to multi-objective optimization.
 * Starting from only one solution, it produces a set of new non-dominated solutions.
 */
template < class EOT > class moeoMOLS:public eoBF < const EOT &, moeoArchive < EOT > &,
  void >
{
};

#endif /*MOEOMOLS_H_ */
