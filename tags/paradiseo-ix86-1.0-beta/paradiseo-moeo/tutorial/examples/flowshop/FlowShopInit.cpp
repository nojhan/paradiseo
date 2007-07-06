// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopInit.cpp
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#include <FlowShopInit.h>


FlowShopInit::FlowShopInit(unsigned int _N) : N(_N)
{}


void FlowShopInit::operator()(FlowShop & _flowshop)
{
    // scheduling vector
    std::vector<unsigned int> scheduling(N);
    // initialisation of possible values
    std::vector<unsigned int> possibles(N);
    for (unsigned int i=0 ; i<N ; i++)
        possibles[i] = i;
    // random initialization
    unsigned int rInd;     // random index
    for (unsigned int i=0; i<N; i++)
    {
        rInd = (unsigned int) rng.uniform(N-i);
        scheduling[i] = possibles[rInd];
        possibles[rInd] = possibles[N-i-1];
    }
    _flowshop.resize(N);
    _flowshop.value(scheduling);
    _flowshop.invalidate();	   // IMPORTANT in case the _genotype is old
}
