// "node.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __node_h
#define __node_h

#include <string>
#include <cassert>

extern int getNodeRank (); /* It gives the rank of the calling process */

extern int getNumberOfNodes (); /* It gives the size of the environment (Total number of nodes) */

extern int getRankFromName (const std :: string & __name); /* It gives the rank of the process
							      expressed by its name */

extern void initNode (int * __argc, char * * * __argv);

#endif
