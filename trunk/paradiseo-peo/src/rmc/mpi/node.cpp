// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "node.cpp"

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

