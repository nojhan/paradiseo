// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; messent-column: 35; -*-

// "rmc.cpp"

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

#include "send.h"
#include "worker.h"
#include "schema.h"
#include "comm.h"
#include "scheduler.h"
#include "../../core/peo_debug.h"

static std :: vector <pthread_t *> ll_threads; /* Low level threads */

void runRMC () {

  /* Worker(s) ? */
  for (unsigned i = 0; i < my_node -> num_workers; i ++) 
    addThread (new Worker, ll_threads);

  wakeUpCommunicator ();
}

void initRMC (int & __argc, char * * & __argv) {

  /* Communication */
  initCommunication ();
  addThread (new Communicator (& __argc, & __argv), ll_threads);
  waitNodeInitialization ();
  initSending ();

  /* Scheduler */
  if (isScheduleNode ())
    initScheduler ();

  ///
}

void finalizeRMC () {

  printDebugMessage ("before join threads RMC");
  joinThreads (ll_threads);
  printDebugMessage ("after join threads RMC");
}
