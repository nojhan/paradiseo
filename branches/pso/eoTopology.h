// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTopology.h
// (c) OPAC 2007
/*
    

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOTOPOLOGY_H_
#define EOTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoNeighborhood.h>
//-----------------------------------------------------------------------------


/**  
 * Define the abstract class for a swarm optimization topology.
 */
template < class POT > class eoTopology:public eoPop < POT >
{
public:
   
    virtual void setup(const eoPop<POT> & _pop)=0;
    virtual void update(POT & ,unsigned _indice)=0;
    virtual POT & best (unsigned ) = 0;
    virtual void printOn(){}
};


#endif /*EOTOPOLOGY_H_ */








