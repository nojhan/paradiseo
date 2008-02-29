/*
* <rmc.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
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

#include "send.h"
#include "worker.h"
#include "schema.h"
#include "comm.h"
#include "scheduler.h"
#include "../../core/peo_debug.h"

static std :: vector <pthread_t *> ll_threads; /* Low level threads */
static std :: vector <Worker *> worker_threads; /* Worker threads */
static Communicator* communicator_thread = NULL; /* Communicator thread */


void runRMC ()
{

  /* Worker(s) ? */
  for (unsigned i = 0; i < my_node -> num_workers; i ++)
    {
      worker_threads.push_back (new Worker);
      addThread (worker_threads.back(), ll_threads);
    }

  wakeUpCommunicator ();
}

void initRMC (int & __argc, char * * & __argv)
{

  /* Communication */
  initCommunication ();
  communicator_thread = new Communicator (& __argc, & __argv);
  addThread (communicator_thread, ll_threads);
  waitNodeInitialization ();
  initSending ();

  /* Scheduler */
  if (isScheduleNode ())
    initScheduler ();
}

void finalizeRMC ()
{

  printDebugMessage ("before join threads RMC");

  joinThreads (ll_threads);
  for (unsigned i = 0; i < worker_threads.size(); i++ )
    {
      delete worker_threads [i];
    }
  worker_threads.clear ();
  delete communicator_thread;

  printDebugMessage ("after join threads RMC");
}
