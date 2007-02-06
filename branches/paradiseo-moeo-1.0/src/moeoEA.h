// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoEA.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOEA_H_
#define MOEOEA_H_

#include <eoAlgo.h>

/**
 * Abstract class for multi-objective evolutionary algorithms
 */
template < class MOEOT >
class moeoEA : public eoAlgo < MOEOT > {};


#endif /*MOEOEA_H_*/
