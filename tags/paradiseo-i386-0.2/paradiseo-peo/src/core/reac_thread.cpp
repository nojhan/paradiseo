// "reac_thread.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "reac_thread.h"

static bool the_end = false;

static std :: vector <ReactiveThread *> reac_threads;

ReactiveThread :: ReactiveThread () {

  reac_threads.push_back (this);
  sem_init (& sem, 0, 0);
}

void ReactiveThread :: sleep () {

  sem_wait (& sem);	
}

void ReactiveThread :: wakeUp () {

  sem_post (& sem);	
}

void stopReactiveThreads () {

  the_end = true;
  for (unsigned i = 0; i < reac_threads.size (); i ++)
    reac_threads [i] -> wakeUp  ();	
}
