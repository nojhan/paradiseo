// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopObjectiveVectorTraits.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOBJECTIVEVECTORTRAITS_H_
#define FLOWSHOPOBJECTIVEVECTORTRAITS_H_

#include <core/moeoObjectiveVectorTraits.h>

/**
 * Definition of the objective vector traits for multi-objective flow-shop problems
 */
class FlowShopObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:

    /**
     * Returns true if the _ith objective have to be minimzed
     * @param _i index of the objective
     */
    static bool minimizing (int _i);


    /**
     * Returns true if the _ith objective have to be maximzed
     * @param _i index of the objective
     */
    static bool maximizing (int _i);


    /**
     * Returns the number of objectives
     */
    static unsigned int nObjectives ();

};

#endif /*FLOWSHOPOBJECTIVEVECTORTRAITS_H_*/
