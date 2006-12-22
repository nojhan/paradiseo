
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "worker.cpp"

// (c) OPAC Team, LIFL, August 2005

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
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
