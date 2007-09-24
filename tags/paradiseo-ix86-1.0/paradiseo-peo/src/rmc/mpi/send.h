// "send.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __send_h
#define __send_h

#include "../../core/communicable.h"

extern void initSending ();

extern void send (Communicable * __comm, int __to, int __tag);

extern void sendToAll (Communicable * __comm, int __tag);

extern void sendMessages ();

#endif
