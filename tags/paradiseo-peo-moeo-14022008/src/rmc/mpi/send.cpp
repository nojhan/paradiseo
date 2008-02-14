/*
* <send.cpp>
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

#include <mpi.h>
#include <semaphore.h>
#include <queue>

#include "tags.h"
#include "comm.h"
#include "worker.h"
#include "scheduler.h"
#include "mess.h"
#include "node.h"
#include "../../core/cooperative.h"
#include "../../core/peo_debug.h"

#define TO_ALL -1


typedef struct
  {

    Communicable * comm;
    int to;
    int tag;

  }
SEND_REQUEST;


static std :: queue <SEND_REQUEST> mess;

static sem_t sem_send;

static bool contextInitialized = false;


void initSending ()
{

  static bool initializedSemaphore = false;

  mess = std :: queue <SEND_REQUEST> ();

  if (initializedSemaphore)
    {
      sem_destroy(& sem_send);
    }

  sem_init (& sem_send, 0, 1);
  initializedSemaphore = true;

  contextInitialized = false;
}

void send (Communicable * __comm, int __to, int __tag)
{

  SEND_REQUEST req;
  req.comm = __comm;
  req.to = __to;
  req.tag = __tag;

  sem_wait (& sem_send);
  mess.push (req);
  sem_post (& sem_send);
  wakeUpCommunicator ();
}

void sendToAll (Communicable * __comm, int __tag)
{

  send (__comm, TO_ALL, __tag);
}

extern void initializeContext ();

void sendMessages ()
{

  if (! contextInitialized)
    {
      contextInitialized = true;
      initializeContext();
    }

  sem_wait (& sem_send);

  while (! mess.empty ())
    {

      SEND_REQUEST req = mess.front ();

      Communicable * comm = req.comm;

      initMessage ();

      switch (req.tag)
        {

        case RUNNER_STOP_TAG:
          dynamic_cast <Runner *> (comm) -> packTermination ();
          dynamic_cast <Runner *> (comm) -> notifySendingTermination ();
          break;

        case COOP_TAG:
          dynamic_cast <Cooperative *> (comm) -> pack ();
          dynamic_cast <Cooperative *> (comm) -> notifySending ();
          break;

        case SYNCHRONIZE_REQ_TAG:
          dynamic_cast <Cooperative *> (comm) -> packSynchronizeReq ();
          dynamic_cast <Cooperative *> (comm) -> notifySendingSyncReq ();
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
