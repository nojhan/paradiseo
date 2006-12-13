// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "reac_thread.cpp"

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
