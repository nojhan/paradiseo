// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEvalFunc.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOEVALFUNC_H_
#define MOEOEVALFUNC_H_

#include <eoEvalFunc.h>

/*
 * Functor that evaluates one MOEO by setting all its objective values.
 */
template < class MOEOT >
class moeoEvalFunc : public eoEvalFunc< MOEOT > {};

#endif /*MOEOEVALFUNC_H_*/
