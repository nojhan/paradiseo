// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "recv.cpp"

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

#include "comm.h"
#include "tags.h"
#include "worker.h"
#include "scheduler.h"
#include "mess.h"
#include "node.h"
#include "../../runner.h"
#include "../../coop.h"
#include "../../peo_debug.h"

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
