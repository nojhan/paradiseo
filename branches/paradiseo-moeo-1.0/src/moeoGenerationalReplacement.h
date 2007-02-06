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

#include <eoGenerationalReplacement.h>
#include <moeoGenerationalReplacement.h>

/**
 * Generational replacement: only the new individuals are preserved
 */
template < class MOEOT >
class moeoGenerationalReplacement : public moeoReplacement < MOEOT >, public eoGenerationalReplacement < MOEOT > {};

#endif /*MOEOGENERATIONALREPLACEMENT_H_*/
