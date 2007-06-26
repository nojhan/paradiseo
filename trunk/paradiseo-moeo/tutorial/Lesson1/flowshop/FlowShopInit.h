// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShopInit.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOPINIT_H_
#define FLOWSHOPINIT_H_

#include <eoInit.h>
#include <FlowShop.h>

/**
 *  Initialization of a random genotype built by the default constructor of the FlowShop class
 */
class FlowShopInit : public eoInit<FlowShop>
{
public:

    /**
     * Ctor
     * @param _N the number of jobs to schedule
     */
    FlowShopInit(unsigned int _N);


    /**
     * builds a random genotype
     * @param _flowshop a genotype that has been default-constructed
     */
    void operator()(FlowShop & _flowshop);


private:

    /** the number of jobs (size of a scheduling vector) */
    unsigned int N;

};

#endif /*FLOWSHOPINIT_H_*/
