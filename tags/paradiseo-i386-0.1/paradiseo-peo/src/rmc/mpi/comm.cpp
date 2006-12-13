// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "comm.cpp"

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

#include "comm.h"
#include "mess.h"
#include "node.h"
#include "param.h"
#include "../../peo_debug.h"
#include "../../runner.h"
#include "send.h"
#include "recv.h"
#include "scheduler.h"

static sem_t sem_comm_init;

static Communicator * the_thread;

Communicator :: Communicator (int * __argc, char * * * __argv) {

  the_thread = this;  
  initNode  (__argc, __argv);
  loadRMCParameters (* __argc, * __argv);  
  sem_post (& sem_comm_init);
}

void Communicator :: start () {

  while (true) {
    
    /* Zzz Zzz Zzz :-))) */
    sleep ();
    sendMessages ();

    if (! atLeastOneActiveRunner ())     
      break;
    receiveMessages ();    
  }
  waitBuffers ();  
  printDebugMessage ("finalizing");
  MPI_Finalize ();  
}

void initCommunication () {

  sem_init (& sem_comm_init, 0, 0);
}

void waitNodeInitialization () {

  sem_wait (& sem_comm_init);
}

void wakeUpCommunicator () {

  the_thread -> wakeUp ();
}



