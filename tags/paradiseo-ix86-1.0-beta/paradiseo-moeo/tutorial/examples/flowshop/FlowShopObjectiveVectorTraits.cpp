// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopObjectiveVectorTraits.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <FlowShopObjectiveVectorTraits.h>


bool FlowShopObjectiveVectorTraits::minimizing (int _i)
{
    // minimizing both
    return true;
}

bool FlowShopObjectiveVectorTraits::maximizing (int _i)
{
    // minimizing both
    return false;
}

unsigned int FlowShopObjectiveVectorTraits::nObjectives ()
{
    // 2 objectives
    return 2;
}
