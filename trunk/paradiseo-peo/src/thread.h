// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "thread.h"

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
   
   Contact: cahon@lifl.fr
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
