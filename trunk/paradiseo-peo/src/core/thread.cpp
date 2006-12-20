// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "thread.cpp"

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <map>

#include "thread.h"

static std :: vector <Thread *> threads;

unsigned num_act = 0;

Thread :: Thread () {
	
  threads.push_back (this);
  act = false;
}

Thread :: ~ Thread () {

  /* Nothing ! */
}

extern int getNodeRank ();

void Thread :: setActive () {

  if (! act ) {

    act = true;
    num_act ++;
    //    if (getNodeRank () == 1)      
    //   printf ("On passe a %d\n", num_act);
  }
}

void Thread :: setPassive () {

  if (act) {

   act = false;
    num_act --;
    //    if (getNodeRank () == 1)      
    //  printf ("On passe a %d\n", num_act);

  } 
}

bool atLeastOneActiveThread () {

  return num_act;
}

unsigned numberOfActiveThreads () {

  return num_act;
}

static void * launch (void * __arg) {

  Thread * thr = (Thread *) __arg;  
  thr -> start ();
  return 0;
}

void addThread (Thread * __hl_thread, std :: vector <pthread_t *> & __ll_threads) {

  pthread_t * ll_thr = new pthread_t;
  __ll_threads.push_back (ll_thr);
  pthread_create (ll_thr, 0, launch, __hl_thread); 
}

void joinThreads (std :: vector <pthread_t *> & __threads) {

  for (unsigned i = 0; i < __threads.size (); i ++)    
    pthread_join (* __threads [i], 0);  
}
