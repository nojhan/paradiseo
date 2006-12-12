// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

#ifndef eoPopAlgo_h
#define eoPopAlgo_h

#include <eoPop.h>
#include <eoFunctor.h>

/**
   For all "population transforming" algos ... 
 */

template <class EOT> class eoPopAlgo : public eoUF <eoPop <EOT> &, void> {
    
} ;
	
#endif
