// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "coop.cpp"

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

#include "../../coop.h"
#include "send.h"
#include "tags.h"
#include "schema.h"
#include "mess.h"
#include "../../peo_debug.h"

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
