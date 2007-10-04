/* 
* <runner.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sébastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
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
