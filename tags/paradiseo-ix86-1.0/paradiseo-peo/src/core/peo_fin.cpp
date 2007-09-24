// "peo_finalize.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "peo_fin.h"
#include "peo_debug.h"
#include "runner.h"
#include "rmc.h"

void peo :: finalize () {

  printDebugMessage ("waiting for the termination of all threads");

  joinRunners ();

  finalizeRMC ();

  printDebugMessage ("this is the end");
  endDebugging ();
}
