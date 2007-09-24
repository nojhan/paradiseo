// "mess.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __mess_rmc_h
#define __mess_rmc_h

#include "../../core/messaging.h"

extern void initMessage ();

extern void sendMessage (int __to, int __tag);

extern void sendMessageToAll (int __tag);

extern void receiveMessage (int __from, int __tag);

extern void cleanBuffers ();

extern void waitBuffers ();

extern bool probeMessage (int & __src, int & __tag);

extern void waitMessage ();

#endif

