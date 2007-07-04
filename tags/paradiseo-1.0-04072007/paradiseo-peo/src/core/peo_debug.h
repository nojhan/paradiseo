// "peo_debug.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peo_debug_h
#define __peo_debug_h

extern void initDebugging ();

extern void endDebugging ();

extern void setDebugMode (bool __dbg = true); /* (Des)activating the Debugging mode */

extern void printDebugMessage (const char * __mess); /* Print a new message both on the
							standard output and a target
							text-file in a subdirectory) */

#endif
