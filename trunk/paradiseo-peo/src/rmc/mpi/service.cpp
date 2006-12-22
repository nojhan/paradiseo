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
   
   Contact: cahon@lifl.fr
*/

#include "../../core/service.h"
#include "../../core/messaging.h"
#include "node.h"
#include "tags.h"
#include "send.h"
#include "scheduler.h"

void Service :: requestResourceRequest (unsigned __how_many) {

  num_sent_rr = __how_many;
  for (unsigned i = 0; i < __how_many; i ++)
    send (this, my_node -> rk_sched, SCHED_REQUEST_TAG);
}

void Service :: packResourceRequest () {

  SCHED_REQUEST req;
  req.first = getNodeRank ();
  req.second = getKey ();
  //  printf ("demande de ressource pour %d\n", req.second);
  :: pack (req);
}
