// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoObjectiveVectorComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOOBJECTIVEVECTORCOMPARATOR_H_
#define MOEOOBJECTIVEVECTORCOMPARATOR_H_

#include <math.h>
#include <eoFunctor.h>

/**
 * Abstract class allowing to compare 2 objective vectors.
 * The template argument ObjectiveVector have to be a moeoObjectiveVector.
 */
template < class ObjectiveVector >
class moeoObjectiveVectorComparator : public eoBF < const ObjectiveVector &, const ObjectiveVector &, const bool > {};

#endif /*MOEOOBJECTIVEVECTORCOMPARATOR_H_*/
