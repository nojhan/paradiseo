// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FlowShop.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef FLOWSHOP_H_
#define FLOWSHOP_H_

#include <core/moeoVector.h>
#include <FlowShopObjectiveVector.h>

/**
 *  Structure of the genotype for the flow-shop scheduling problem: a vector of unsigned int int.
 */
class FlowShop: public moeoVector < FlowShopObjectiveVector , double , double , unsigned int >
{
public:

    /**
     * class name
     */
    std::string className() const;

};

#endif /*FLOWSHOP_H_*/
