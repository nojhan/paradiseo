// "reac_thread.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef REAC_THREAD_H_
#define REAC_THREAD_H_

#include <semaphore.h>

#include "thread.h"

class ReactiveThread : public Thread {
	
public:

  /* Ctor */
  ReactiveThread ();

  void sleep ();
  
  void wakeUp ();
    
private:

  sem_t sem;
   
};

extern void stopReactiveThreads ();

#endif /*THREAD_H_*/
