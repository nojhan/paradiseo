// "runner.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <vector>

#include "runner.h"
#include "reac_thread.h"
#include "peo_debug.h"
#include "messaging.h"

static unsigned num_act = 0; /* Number of active runners */

static std :: vector <pthread_t *> ll_threads; /* Low-level runner threads */ 

static std :: vector <Runner *> the_runners;

static unsigned num_runners = 0;

Runner :: Runner () {

  id = ++ num_runners;
  the_runners.push_back (this);
  sem_init (& sem_start, 0, 0);
  num_act ++;  
}

extern int getNodeRank ();

extern int getNumberOfNodes ();

void unpackTerminationOfRunner () {
  
  RUNNER_ID id;
  unpack (id);    
  num_act --;
  printDebugMessage ("I'm noticed of the termination of a runner");
  if (! num_act) {
    printDebugMessage ("all the runners have terminated. Now stopping the reactive threads.");
    stopReactiveThreads ();
  }
}

bool atLeastOneActiveRunner () {

  return num_act;
}

RUNNER_ID Runner :: getID () {

  return id;
}

void Runner :: start () {

  setActive ();
  sem_post (& sem_start);
  run ();
  terminate ();
}

void Runner :: notifySendingTermination () {

  /*
  char b [1000];
  sprintf (b, "Il reste encore %d !!!!!!!!!!!!", n);
  printDebugMessage (b);
  */
  printDebugMessage ("je suis informe que tout le monde a recu ma terminaison");
  setPassive ();
  
}

void Runner :: waitStarting () {

  sem_wait (& sem_start);
}

Runner * getRunner (RUNNER_ID __key) {

  return dynamic_cast <Runner *> (getCommunicable (__key));
}

void startRunners () {
  
  /* Runners */
  for (unsigned i = 0; i < the_runners.size (); i ++)
    if (the_runners [i] -> isLocal ()) {
      addThread (the_runners [i], ll_threads);
      the_runners [i] -> waitStarting ();
    }
  printDebugMessage ("launched the parallel runners");
}


void joinRunners () {


  joinThreads (ll_threads);
}
