/* 
* <communicable.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
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

void initCommunicableEnv () {

  key_to_comm.resize (1);
  comm_to_key.clear ();
  Communicable :: num_comm = 0;
}
