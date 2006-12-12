// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "send.cpp"

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

#include <mpi.h>
#include <semaphore.h>
#include <queue>

#include "tags.h"
#include "comm.h"
#include "worker.h"
#include "scheduler.h"
#include "mess.h"
#include "node.h"
#include "../../coop.h"
#include "../../peo_debug.h"

#define TO_ALL -1

typedef struct {

  Communicable * comm;
  int to;
  int tag;

} SEND_REQUEST;
	
static std :: queue <SEND_REQUEST> mess;

static sem_t sem_send;

void initSending () {

  sem_init (& sem_send, 0, 1);
}

void send (Communicable * __comm, int __to, int __tag) {

  SEND_REQUEST req;  
  req.comm = __comm;
  req.to = __to;
  req.tag = __tag;

  sem_wait (& sem_send);
  mess.push (req);
  sem_post (& sem_send);
  wakeUpCommunicator ();
}

void sendToAll (Communicable * __comm, int __tag) {
  
  send (__comm, TO_ALL, __tag);
}

void sendMessages () {

  sem_wait (& sem_send);

  while (! mess.empty ()) {
    
    SEND_REQUEST req = mess.front ();
    /*
    char b [1000];
    sprintf (b, "traitement send %d\n", req.tag);
    printDebugMessage (b);
    */
    
    Communicable * comm = req.comm;

    initMessage ();

    switch (req.tag) {

    case RUNNER_STOP_TAG:
      dynamic_cast <Runner *> (comm) -> packTermination ();            
      dynamic_cast <Runner *> (comm) -> notifySendingTermination ();            
      break;

    case COOP_TAG:
      dynamic_cast <Cooperative *> (comm) -> pack ();      
      dynamic_cast <Cooperative *> (comm) -> notifySending ();      
      break;
          
    case SCHED_REQUEST_TAG:
      dynamic_cast <Service *> (comm) -> packResourceRequest ();
      dynamic_cast <Service *> (comm) -> notifySendingResourceRequest ();            
      break;

    case TASK_RESULT_TAG:
      dynamic_cast <Worker *> (comm) -> packResult ();
      dynamic_cast <Worker *> (comm) -> notifySendingResult ();
      break;

    case TASK_DONE_TAG:
      dynamic_cast <Worker *> (comm) -> packTaskDone ();
      dynamic_cast <Worker *> (comm) -> notifySendingTaskDone ();
      break;
      
    default :
      break;

    };
    
    if (req.to == TO_ALL)
      sendMessageToAll (req.tag);
    else
      sendMessage (req.to, req.tag);
    mess.pop ();
  }

  sem_post (& sem_send);  
}
