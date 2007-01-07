// "schema.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __schema_h
#define __schema_h

#include <string>
#include <vector>
#include <cassert>

#include "../../core/runner.h"

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
