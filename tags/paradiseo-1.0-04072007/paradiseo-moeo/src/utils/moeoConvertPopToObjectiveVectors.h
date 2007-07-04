// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoConvertPopToObjectiveVectors.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOPOPTOOBJECTIVEVECTORS_H_
#define MOEOPOPTOOBJECTIVEVECTORS_H_

#include <vector>
#include <eoFunctor.h>

/**
 * Functor allowing to get a vector of objective vectors from a population
 */
template < class MOEOT, class ObjectiveVector = typename MOEOT::ObjectiveVector >
class moeoConvertPopToObjectiveVectors : public eoUF < const eoPop < MOEOT >, const std::vector < ObjectiveVector > >
{
public:

    /**
     * Returns a vector of the objective vectors from the population _pop
     * @param _pop the population
     */
    const std::vector < ObjectiveVector > operator()(const eoPop < MOEOT > _pop)
    {
        std::vector < ObjectiveVector > result;
        result.resize(_pop.size());
        for (unsigned int i=0; i<_pop.size(); i++)
        {
            result.push_back(_pop[i].objectiveVector());
        }
        return result;
    }

};

#endif /*MOEOPOPTOOBJECTIVEVECTORS_H_*/
