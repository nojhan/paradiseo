// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "communicable.h"

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
