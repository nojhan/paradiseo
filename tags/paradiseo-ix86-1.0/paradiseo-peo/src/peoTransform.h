// "peoTransform.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peoTransform_h
#define __peoTransform_h

#include <eoTransform.h>

#include "core/service.h"

//! Interface class for constructing more complex transformation operators.

//! The peoTransform class acts only as an interface for creating transform operators - for an example
//! please refer to the <b>peoSeqTransform</b> and the <b>peoParaSGATransform</b> classes.
template< class EOT > class peoTransform : public Service, public eoTransform< EOT > {

};


#endif
