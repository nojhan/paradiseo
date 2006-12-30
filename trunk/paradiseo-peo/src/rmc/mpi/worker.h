// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "worker.h"

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

#ifndef __worker_h
#define __worker_h

#include "../../core/communicable.h"
#include "../../core/reac_thread.h"
#include "../../core/service.h"

typedef unsigned WORKER_ID; 

class Worker : public Communicable, public ReactiveThread {

public : 

  Worker ();

  void start ();

  void packResult ();

  void unpackData ();

  void packTaskDone (); 

  void notifySendingResult ();

  void notifySendingTaskDone ();
  
  void setSource (int __rank);
  
private :

  WORKER_ID id;
  SERVICE_ID serv_id;
  Service * serv;
  int src;

  bool toto;
};

extern Worker * getWorker (WORKER_ID __key);

#endif
