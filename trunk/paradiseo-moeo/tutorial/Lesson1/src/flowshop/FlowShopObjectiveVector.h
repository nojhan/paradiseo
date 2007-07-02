// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopObjectiveVector.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOBJECTIVEVECTOR_H_
#define FLOWSHOPOBJECTIVEVECTOR_H_

#include <core/moeoRealObjectiveVector.h>
#include <FlowShopObjectiveVectorTraits.h>

/**
 * Definition of the objective vector for multi-objective flow-shop problems: a vector of doubles
 */
typedef moeoRealObjectiveVector < FlowShopObjectiveVectorTraits > FlowShopObjectiveVector;

#endif /*FLOWSHOPOBJECTIVEVECTOR_H_*/
