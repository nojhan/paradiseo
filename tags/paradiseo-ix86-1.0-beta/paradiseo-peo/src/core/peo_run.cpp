// "peo_run.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "peo_init.h"
#include "peo_run.h"
#include "rmc.h"
#include "runner.h"

void peo :: run () {
  
  startRunners ();

  runRMC ();
}
