// "comm.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/


#include <mpi.h>

#include "comm.h"
#include "mess.h"
#include "node.h"
#include "param.h"
#include "../../core/peo_debug.h"
#include "../../core/runner.h"
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



