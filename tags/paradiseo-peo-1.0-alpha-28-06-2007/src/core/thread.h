// "thread.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef THREAD_H_
#define THREAD_H_

#include <vector>

/* A high-level thread */

class Thread {
	
public:

  /* Ctor */
  Thread ();

  /* Dtor */
  virtual ~ Thread ();
  
  /* Go ! */
  virtual void start () = 0;

  void setActive ();/* It means the current process is going to send messages soon */
  void setPassive ();/* The current process is not going to perform send operations
			(but it may receive messages) */

private :
  
  bool act;
};

extern void addThread (Thread * __hl_thread, std :: vector <pthread_t *> & __ll_threads);

extern void joinThreads (std :: vector <pthread_t *> & __ll_threads);

extern bool atLeastOneActiveThread (); /* It returns 'true' iff at least one process is going
				      to send messages */
  
extern unsigned numberOfActiveThreads ();


#endif /*THREAD_H_*/
