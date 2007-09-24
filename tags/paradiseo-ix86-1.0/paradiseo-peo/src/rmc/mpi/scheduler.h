// "scheduler.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __scheduler_h
#define __scheduler_h

#include <utility>

#include "schema.h"
#include "worker.h"

typedef std :: pair <RANK_ID, WORKER_ID> SCHED_RESOURCE;

typedef std :: pair <RANK_ID, SERVICE_ID> SCHED_REQUEST;

/* Initializing the list of available workers */
extern void initScheduler ();

/* Processing a resource request from a service */
extern void unpackResourceRequest ();

/* Being known a worker is now idle :-) */
extern void unpackTaskDone (); 

extern bool allResourcesFree ();

#endif
