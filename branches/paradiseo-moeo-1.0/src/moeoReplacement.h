// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoReplacement.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOREPLACEMENT_H_
#define MOEOREPLACEMENT_H_

#include <eoReplacement.h>

/**
 * Replacement strategy for multi-objective optimization.
 */
template < class MOEOT >
class moeoReplacement : public eoReplacement < MOEOT >
{};

#endif /*MOEOREPLACEMENT_H_*/
