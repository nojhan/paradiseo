// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "service.h"

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

#ifndef __service_h
#define __service_h

#include "communicable.h"
#include "thread.h"

typedef unsigned SERVICE_ID;

class Service : public Communicable {

public :

  void setOwner (Thread & __owner);
  
  Thread * getOwner (); 

  void requestResourceRequest (unsigned __how_many = 1);
  void packResourceRequest ();

  virtual void packData ();
  virtual void unpackData ();

  virtual void execute ();
  
  virtual void packResult ();
  virtual void unpackResult ();

  virtual void notifySendingData ();
  virtual void notifySendingResourceRequest ();
  virtual void notifySendingAllResourceRequests ();

private :

  Thread * owner; /* Owner thread (i.e. 'uses' that service) */ 

  unsigned num_sent_rr; /* Number of RR not really sent (i.e. still in the sending queue)*/

};

extern Service * getService (SERVICE_ID __key); 

#endif
