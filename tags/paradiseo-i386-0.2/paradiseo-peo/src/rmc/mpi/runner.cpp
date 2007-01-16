// "runner.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "../../core/messaging.h"
#include "../../core/runner.h"
#include "node.h"
#include "send.h"
#include "tags.h"
#include "schema.h"

bool Runner :: isLocal () {

  for (unsigned i = 0; i < my_node -> id_run.size (); i ++)
    if (my_node -> id_run [i] == id)
      return true;
  return false;
}

void Runner :: packTermination () {

  pack (id);
}

void Runner :: terminate () {

  sendToAll (this, RUNNER_STOP_TAG);     
}

