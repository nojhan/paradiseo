// "service.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
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
