// "comm.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __comm_mpi_h
#define __comm_mpi_h

#include "../../core/communicable.h"
#include "../../core/reac_thread.h"

class Communicator : public ReactiveThread {

public :
  
  /* Ctor */
  Communicator (int * __argc, char * * * __argv);

  void start ();
};

extern void initCommunication ();

extern void waitNodeInitialization ();

extern void wakeUpCommunicator ();

#endif
