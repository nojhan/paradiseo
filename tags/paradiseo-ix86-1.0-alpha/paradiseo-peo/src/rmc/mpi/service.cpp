// "service.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
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
