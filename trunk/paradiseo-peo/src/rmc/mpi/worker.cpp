/* 
* <worker.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
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
  
  recvAndCompleted = false;
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
  recvAndCompleted = true;
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

    if (recvAndCompleted) {
      send (this, my_node -> rk_sched, TASK_DONE_TAG);  
      recvAndCompleted = false;
    }
    else {

      printDebugMessage ("executing the task.");
      serv -> execute ();   
      send (this, src, TASK_RESULT_TAG);    
    }
  }
}

void initWorkersEnv () {

  key_to_worker.resize (1);
}
