// "communicable.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __communicable_h
#define __communicable_h

#include <semaphore.h>

typedef unsigned COMM_ID;

class Communicable {

public :

  Communicable ();
  
  virtual ~ Communicable ();

  COMM_ID getKey ();  

  void lock (); /* It suspends the current process if the semaphore is locked */
  void unlock (); /* It unlocks the shared semaphore */

  void stop (); /* It suspends the current process */
  void resume (); /* It resumes ___________ */
  
protected :

  COMM_ID key;

  sem_t sem_lock;
  
  sem_t sem_stop;

  static unsigned num_comm;
};

extern Communicable * getCommunicable (COMM_ID __key); 

//extern COMM_ID getKey (const Communicable * __comm);

#endif
