// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "runner.h"

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

#ifndef __runner_h
#define __runner_h

#include <eoFunctor.h>

#include "communicable.h"
#include "thread.h"

typedef unsigned RUNNER_ID;

class Runner : public Communicable, public Thread {

public :

  Runner ();

  void start ();

  void waitStarting ();

  bool isLocal ();

  void terminate ();

  virtual void run () = 0;
  
  RUNNER_ID getID (); 

  void packTermination ();

  void notifySendingTermination ();

private :

  sem_t sem_start;

  unsigned id;
};

extern bool atLeastOneActiveRunner ();

extern void unpackTerminationOfRunner ();

extern Runner * getRunner (RUNNER_ID __key); 

extern void startRunners ();

extern void joinRunners ();

#endif
