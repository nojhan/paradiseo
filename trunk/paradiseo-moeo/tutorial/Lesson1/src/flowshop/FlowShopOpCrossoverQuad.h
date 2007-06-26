// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpCrossoverQuad.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOPCROSSOVERQUAD_H_
#define FLOWSHOPOPCROSSOVERQUAD_H_

#include <eoOp.h>
#include <FlowShop.h>

/**
 * Quadratic crossover operator for flow-shop (modify the both genotypes)
 */
class FlowShopOpCrossoverQuad : public eoQuadOp < FlowShop >
{
public:

    /**
     * the class name (used to display statistics)
     */
    std::string className() const;


    /**
     * eoQuad crossover - _flowshop1 and _flowshop2 are the (future) offspring, i.e. _copies_ of the parents
     * @param _flowshop1 the first parent
     * @param _flowshop2 the second parent
     */
    bool operator()(FlowShop & _flowshop1, FlowShop & _flowshop2);


private:

    /**
     * generation of an offspring by a 2 points crossover
     * @param _parent1 the first parent
     * @param _parent2 the second parent
     * @param _point1 the first point
     * @param _point2 the second point
     */
    FlowShop generateOffspring(const FlowShop & _parent1, const FlowShop & _parent2, unsigned int _point1, unsigned int _point2);

};

#endif /*FLOWSHOPOPCROSSOVERQUAD_H_*/
