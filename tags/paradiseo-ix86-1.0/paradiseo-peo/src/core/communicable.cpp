// "comm.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
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



