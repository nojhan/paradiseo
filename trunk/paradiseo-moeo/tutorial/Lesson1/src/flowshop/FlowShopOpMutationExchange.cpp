// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpCrossoverQuad.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <FlowShopOpMutationExchange.h>


std::string FlowShopOpMutationExchange::className() const
{
    return "FlowShopOpMutationExchange";
}


bool FlowShopOpMutationExchange::operator()(FlowShop & _flowshop)
{
    bool isModified;
    FlowShop result = _flowshop;
    // computation of the 2 random points
    unsigned int point1, point2;
    do
    {
        point1 = rng.random(result.size());
        point2 = rng.random(result.size());
    } while (point1 == point2);
    // swap
    std::swap (result[point1], result[point2]);
    // update (if necessary)
    if (result != _flowshop)
    {
        // update
        _flowshop.value(result);
        // the genotype has been modified
        isModified = true;
    }
    else
    {
        // the genotype has not been modified
        isModified = false;
    }
    // return 'true' if the genotype has been modified
    return isModified;
}
