// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopFitness.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2006
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef _FlowShopFitness_h
#define _FlowShopFitness_h

#include <eoParetoFitness.h>


/**
 * definition of the fitness for multi-objective flow-shop problems
 */
typedef eoParetoFitness < eoVariableParetoTraits > FlowShopFitness;

#endif
