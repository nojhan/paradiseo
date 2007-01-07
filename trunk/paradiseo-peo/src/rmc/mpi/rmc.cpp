// "rmc.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "send.h"
#include "worker.h"
#include "schema.h"
#include "comm.h"
#include "scheduler.h"
#include "../../core/peo_debug.h"

static std :: vector <pthread_t *> ll_threads; /* Low level threads */

void runRMC () {

  /* Worker(s) ? */
  for (unsigned i = 0; i < my_node -> num_workers; i ++) 
    addThread (new Worker, ll_threads);

  wakeUpCommunicator ();
}

void initRMC (int & __argc, char * * & __argv) {

  /* Communication */
  initCommunication ();
  addThread (new Communicator (& __argc, & __argv), ll_threads);
  waitNodeInitialization ();
  initSending ();

  /* Scheduler */
  if (isScheduleNode ())
    initScheduler ();

  ///
}

void finalizeRMC () {

  printDebugMessage ("before join threads RMC");
  joinThreads (ll_threads);
  printDebugMessage ("after join threads RMC");
}
