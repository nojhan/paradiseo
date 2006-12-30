// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "scheduler.h"

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
