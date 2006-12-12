// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "schema.h"

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

#ifndef __schema_h
#define __schema_h

#include <string>
#include <vector>
#include <cassert>

#include "../../runner.h"

typedef int RANK_ID;

struct Node {
  
  RANK_ID rk; /* Rank */
  std :: string name; /* Host name */
  unsigned num_workers; /* Number of parallel workers */
  int rk_sched; /* rank of the scheduler */
  std :: vector <RUNNER_ID> id_run; /* List of runners */
};

extern std :: vector <Node> the_schema;

extern Node * my_node;

extern void loadSchema (const char * __filename);

extern RANK_ID getRankOfRunner (RUNNER_ID __key);

extern bool isScheduleNode ();

#endif
