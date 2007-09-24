// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpMutationShift.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPOPMUTATIONSHIFT_H_
#define FLOWSHOPOPMUTATIONSHIFT_H_

#include <eoOp.h>
#include <FlowShop.h>

/**
 * Shift mutation operator for flow-shop
 */
class FlowShopOpMutationShift : public eoMonOp < FlowShop >
{
public:

    /**
     * the class name (used to display statistics)
     */
    std::string className() const;


    /**
     * modifies the parent with a shift mutation
     * @param _flowshop the parent genotype (will be modified)
     */
    bool operator()(FlowShop & _flowshop);

};

#endif /*FLOWSHOPOPMUTATIONSHIFT_H_*/
