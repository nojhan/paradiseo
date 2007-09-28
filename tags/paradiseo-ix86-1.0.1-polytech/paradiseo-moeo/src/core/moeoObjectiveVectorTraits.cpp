// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVectorTraits.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <core/moeoObjectiveVectorTraits.h>

// The static variables of the moeoObjectiveVectorTraits class need to be allocated
unsigned int moeoObjectiveVectorTraits::nObj;
std::vector < bool > moeoObjectiveVectorTraits::bObj;
