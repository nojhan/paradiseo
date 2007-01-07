// "worker.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <vector>

#include "tags.h"
#include "send.h"
#include "node.h"
#include "schema.h"
#include "worker.h"
#include "mess.h"
#include "../../core/peo_debug.h"

static std :: vector <Worker *> key_to_worker (1); /* Vector of registered workers */

Worker * getWorker (WORKER_ID __key) {

  return key_to_worker [__key];  
}

Worker :: Worker () {
  
  toto = false;
  id = key_to_worker.size ();
  key_to_worker.push_back (this);
}

void Worker :: packResult () {
  
  pack (serv_id);
  serv -> packResult ();    
}

void Worker :: unpackData () {

  printDebugMessage ("unpacking the ID. of the service.");
  unpack (serv_id);
  serv = getService (serv_id); 
  printDebugMessage ("found the service.");
  serv -> unpackData (); 
  printDebugMessage ("unpacking the data.");
  setActive ();
}

void Worker :: packTaskDone () {

  pack (getNodeRank ());
  pack (id);
}

void Worker :: notifySendingResult () {

  /* Notifying the scheduler of the termination */
  toto = true;
  wakeUp ();
}

void Worker :: notifySendingTaskDone () {

  setPassive ();
}
  
void Worker :: setSource (int __rank) {

  src = __rank;
}

void Worker :: start () {

  while (true) {
    
    sleep (); 

    if (! atLeastOneActiveRunner ())
      break;
    
    if (toto) {
      send (this, my_node -> rk_sched, TASK_DONE_TAG);  
      toto = false;
    }
    else {

      printDebugMessage ("executing the task.");
      serv -> execute ();   
      send (this, src, TASK_RESULT_TAG);    
    }
  }
}
