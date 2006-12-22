// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "cooperative.h"

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

#ifndef __cooperative_h
#define __cooperative_h

#include "communicable.h"
#include "runner.h"

typedef unsigned COOP_ID;

class Cooperative : public Communicable {

public :

  Runner * getOwner ();

  void setOwner (Runner & __runner);

  virtual void pack () = 0;
  
  virtual void unpack () = 0;

  void send (Cooperative * __coop); 

  virtual void notifySending ();

private :

  Runner * owner;

};

extern Cooperative * getCooperative (COOP_ID __key); 

#endif
