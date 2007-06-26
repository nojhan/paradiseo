// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpCrossoverQuad.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOPMUTATIONEXCHANGE_H_
#define FLOWSHOPOPMUTATIONEXCHANGE_H_

#include <eoOp.h>
#include <FlowShop.h>

/**
 * Exchange mutation operator for the flow-shop
 */
class FlowShopOpMutationExchange : public eoMonOp<FlowShop>
{
public:

    /**
     * the class name (used to display statistics)
     */
    std::string className() const;


    /**
     * modifies the parent with an exchange mutation
     * @param _flowshop the parent genotype (will be modified)
     */
    bool operator()(FlowShop & _flowshop);

};

#endif /*FLOWSHOPOPMUTATIONEXCHANGE_H_*/
