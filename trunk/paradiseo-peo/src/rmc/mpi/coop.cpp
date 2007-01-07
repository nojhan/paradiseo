// "coop.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "../../core/cooperative.h"
#include "send.h"
#include "tags.h"
#include "schema.h"
#include "mess.h"
#include "../../core/peo_debug.h"

Runner * Cooperative :: getOwner () {

  return owner;
}

void Cooperative :: setOwner (Runner & __runner) {

  owner = & __runner;
}

void Cooperative :: send (Cooperative * __coop) {

  :: send (this, getRankOfRunner (__coop -> getOwner () -> getID ()), COOP_TAG);   
  //  stop ();
}

Cooperative * getCooperative (COOP_ID __key) {

  return dynamic_cast <Cooperative *> (getCommunicable (__key));
}

void Cooperative :: notifySending () {

  //getOwner -> setPassive ();
  //  resume ();
  //  printDebugMessage (b);
}
