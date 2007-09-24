// "worker.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __worker_h
#define __worker_h

#include "../../core/communicable.h"
#include "../../core/reac_thread.h"
#include "../../core/service.h"

typedef unsigned WORKER_ID; 

class Worker : public Communicable, public ReactiveThread {

public : 

  Worker ();

  void start ();

  void packResult ();

  void unpackData ();

  void packTaskDone (); 

  void notifySendingResult ();

  void notifySendingTaskDone ();
  
  void setSource (int __rank);
  
private :

  WORKER_ID id;
  SERVICE_ID serv_id;
  Service * serv;
  int src;

  bool toto;
};

extern Worker * getWorker (WORKER_ID __key);

#endif
