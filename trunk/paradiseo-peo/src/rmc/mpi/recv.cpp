/* 
* <recv.cpp>
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

#include "comm.h"
#include "tags.h"
#include "worker.h"
#include "scheduler.h"
#include "mess.h"
#include "node.h"
#include "../../core/runner.h"
#include "../../core/cooperative.h"
#include "../../core/peo_debug.h"

void receiveMessages () {

  cleanBuffers ();
    
  do {

    if (! atLeastOneActiveThread ()) {
      //      printDebugMessage ("debut wait");
      waitMessage ();
      //printDebugMessage ("fin wait");
    }
    
    int src, tag;

    while (probeMessage (src, tag)) {
      
      receiveMessage (src, tag);
      initMessage ();
      /*
      char b [1000];
      sprintf (b, "traitement recv %d\n", tag);
      printDebugMessage (b);
      */
      
      switch (tag) {
	
      case RUNNER_STOP_TAG:	
	unpackTerminationOfRunner ();	
	wakeUpCommunicator ();
	break;
      
      case COOP_TAG:
	//	printDebugMessage ("reception de message de cooperation");
	COOP_ID coop_id;
	unpack (coop_id);
	getCooperative (coop_id) -> unpack ();
	break;

      case SCHED_REQUEST_TAG:	
	unpackResourceRequest ();
	break;
	
      case SCHED_RESULT_TAG:
	{	  
	  /* Unpacking the resource */
	  SERVICE_ID serv_id;
	  unpack (serv_id);
	  Service * serv = getService (serv_id);
	  int dest;
	  unpack (dest);
	  WORKER_ID worker_id;
 	  unpack (worker_id);

	  /* Going back ... */
	  initMessage ();
	  pack (worker_id);
	  pack (serv_id); 
	  serv -> packData ();
	  serv -> notifySendingData ();
	  sendMessage (dest, TASK_DATA_TAG);
	  break;
	}

      case TASK_DATA_TAG:
      {
        WORKER_ID worker_id;
	unpack (worker_id);		
	Worker * worker = getWorker (worker_id);
	worker -> setSource (src);
	worker -> unpackData ();
	worker -> wakeUp ();
	break; 
      }
      
      case TASK_RESULT_TAG:
	{
	  SERVICE_ID serv_id;
	  unpack (serv_id);
	  Service * serv = getService (serv_id);
	  serv -> unpackResult ();
	  break;
	}

      case TASK_DONE_TAG:
	unpackTaskDone ();
	break;

      default:
	;
      };
    }
        
  } while (! atLeastOneActiveThread () && atLeastOneActiveRunner () /*&& ! allResourcesFree ()*/);
}
