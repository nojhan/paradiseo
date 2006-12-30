// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "service.cpp"

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

#include "service.h"

void Service :: setOwner (Thread & __owner) {

  owner = & __owner;
}
  
Thread * Service :: getOwner () {

  return owner;
}

Service * getService (SERVICE_ID __key) {

  return dynamic_cast <Service *> (getCommunicable (__key));
}

void Service :: notifySendingData () {

}
void Service :: notifySendingResourceRequest () {

  num_sent_rr --;
  if (! num_sent_rr)
    notifySendingAllResourceRequests ();
}

void Service :: notifySendingAllResourceRequests () {

}

void Service :: packData () {

}

void Service :: unpackData () {

}

void Service :: execute () {

}
  
void Service :: packResult () {

}

void Service :: unpackResult () {

}
