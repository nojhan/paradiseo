// "runner.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __runner_h
#define __runner_h

#include <eoFunctor.h>

#include "communicable.h"
#include "thread.h"

typedef unsigned RUNNER_ID;

class Runner : public Communicable, public Thread {

public :

  Runner ();

  void start ();

  void waitStarting ();

  bool isLocal ();

  void terminate ();

  virtual void run () = 0;
  
  RUNNER_ID getID (); 

  void packTermination ();

  void notifySendingTermination ();

private :

  sem_t sem_start;

  unsigned id;
};

extern bool atLeastOneActiveRunner ();

extern void unpackTerminationOfRunner ();

extern Runner * getRunner (RUNNER_ID __key); 

extern void startRunners ();

extern void joinRunners ();

#endif
