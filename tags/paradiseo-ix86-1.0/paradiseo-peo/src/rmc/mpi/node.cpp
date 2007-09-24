// "node.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <mpi.h>
#include <vector>
#include <map>
#include <string>
#include <cassert>

static int rk, sz; /* Rank & size */

static std :: map <std :: string, int> name_to_rk;

static std :: vector <std :: string> rk_to_name;

int getNodeRank () {

  return rk;
}

int getNumberOfNodes () {

  return sz;
}

int getRankFromName (const std :: string & __name) {
  
  return atoi (__name.c_str ());  
}

void initNode (int * __argc, char * * * __argv) {
  
  int provided;
  MPI_Init_thread (__argc,  __argv, MPI_THREAD_FUNNELED, & provided);  
  assert (provided == MPI_THREAD_FUNNELED); /* The MPI implementation must be multi-threaded.
					       Yet, only one thread performs the comm.
					       operations */
  MPI_Comm_rank (MPI_COMM_WORLD, & rk);   /* Who ? */
  MPI_Comm_size (MPI_COMM_WORLD, & sz);    /* How many ? */

  char names [sz] [MPI_MAX_PROCESSOR_NAME];
  int len;

  /* Processor names */ 
  MPI_Get_processor_name (names [0], & len);   /* Me */  
  MPI_Allgather (names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD); /* Broadcast */
  
  for (int i = 0; i < sz; i ++) {
    rk_to_name.push_back (names [i]);
    name_to_rk [names [i]] = i;
  }
}

