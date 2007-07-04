// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopOpMutationShift.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <FlowShopOpMutationShift.h>


std::string FlowShopOpMutationShift::className() const
{
    return "FlowShopOpMutationShift";
}


bool FlowShopOpMutationShift::operator()(FlowShop & _flowshop)
{
    bool isModified;
    int direction;
    unsigned int tmp;
    FlowShop result = _flowshop;
    // computation of the 2 random points
    unsigned int point1, point2;
    do
    {
        point1 = rng.random(result.size());
        point2 = rng.random(result.size());
    } while (point1 == point2);
    // direction
    if (point1 < point2)
        direction = 1;
    else
        direction = -1;
    // mutation
    tmp = result[point1];
    for (unsigned int i=point1 ; i!=point2 ; i+=direction)
        result[i] = result[i+direction];
    result[point2] = tmp;
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
