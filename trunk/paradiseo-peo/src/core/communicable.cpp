// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "comm.cpp"

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

#include <vector>
#include <map>
#include <cassert>

#include "communicable.h"

static std :: vector <Communicable *> key_to_comm (1); /* Vector of registered cooperators */

static std :: map <const Communicable *, unsigned> comm_to_key; /* Map of registered cooperators */

unsigned Communicable :: num_comm = 0;

Communicable :: Communicable () {

  comm_to_key [this] = key = ++ num_comm;
  key_to_comm.push_back (this);
  sem_init (& sem_lock, 0, 1);
  sem_init (& sem_stop, 0, 0);
}

Communicable :: ~ Communicable () {

}

COMM_ID Communicable :: getKey () {

  return key;
}

Communicable * getCommunicable (COMM_ID __key) {

  assert (__key < key_to_comm.size ());
  return key_to_comm [__key];  
}

COMM_ID getKey (const Communicable * __comm) {
  
  return comm_to_key [__comm];
}

void Communicable :: lock () {

  sem_wait (& sem_lock);
}

void Communicable :: unlock () {

  sem_post (& sem_lock);
}

void Communicable :: stop () {

  sem_wait (& sem_stop);
}

void Communicable :: resume () {

  sem_post (& sem_stop);
}



