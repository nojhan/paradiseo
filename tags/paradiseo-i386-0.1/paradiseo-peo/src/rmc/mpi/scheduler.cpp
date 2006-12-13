// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "sched_thread.cpp"

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

#include <queue>

#include "scheduler.h"
#include "tags.h"
#include "mess.h"
#include "../../peo_debug.h"

static std :: queue <SCHED_RESOURCE> resources; /* Free resources */

static std :: queue <SCHED_REQUEST> requests; /* Requests */

static unsigned initNumberOfRes = 0;

void initScheduler () {
  
  for (unsigned i = 0; i < the_schema.size (); i ++) {
    
    const Node & node = the_schema [i];
    
    if (node.rk_sched == my_node -> rk)      
      for (unsigned j = 0; j < node.num_workers; j ++)
	resources.push (std :: pair <RANK_ID, WORKER_ID> (i, j + 1));    
  }  
  initNumberOfRes = resources.size ();
}

bool allResourcesFree () {

  return resources.size () == initNumberOfRes;
}

static void update () {

  unsigned num_alloc = std :: min (resources.size (), requests.size ());
  
  for (unsigned i = 0; i < num_alloc; i ++) {
    
    SCHED_REQUEST req = requests.front ();
    requests.pop ();
    
    SCHED_RESOURCE res = resources.front ();
    resources.pop ();

    printDebugMessage ("allocating a resource.");    
    initMessage ();
    pack (req.second);
    pack (res);
    sendMessage (req.first, SCHED_RESULT_TAG);
  }  
}

void unpackResourceRequest () {

  printDebugMessage ("queuing a resource request.");
  SCHED_REQUEST req;
  unpack (req);
  requests.push (req);
  update ();
}

void unpackTaskDone () {

  printDebugMessage ("I'm notified a worker is now idle.");
  SCHED_RESOURCE res;
  unpack (res);
  resources.push (res);
  if (resources.size () == initNumberOfRes)
    printDebugMessage ("all the resources are now free.");
  update ();
}
