/*
* <node.cpp>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#include <mpi.h>
#include <vector>
#include <map>
#include <string>
#include <cassert>

#include "mess.h"


class MPIThreadedEnv
  {

  public:

    static void init ( int * __argc, char * * * __argv )
    {

      static MPIThreadedEnv mpiThreadedEnv( __argc, __argv );
    }

    static void finalize ()
    {

      static bool finalizedEnvironment = false;

      if (! finalizedEnvironment )
        {

          MPI_Finalize ();
          finalizedEnvironment = true;
        }
    }

  private:

    /* No instance of this class can be created outside its domain! */
    MPIThreadedEnv ( int * __argc, char * * * __argv )
    {

      static bool MPIThreadedEnvInitialized = false;
      int provided = 1;

      if (! MPIThreadedEnvInitialized)
        {

          MPI_Init_thread (__argc, __argv, MPI_THREAD_FUNNELED, & provided);

          assert (provided == MPI_THREAD_FUNNELED); /* The MPI implementation must be multi-threaded.
          					       Yet, only one thread performs the comm.
          					       operations */
          MPIThreadedEnvInitialized = true;
        }
    }

    ~MPIThreadedEnv()
    {

      finalize ();
    }
  };


static int rk, sz; /* Rank & size */

static std :: map <std :: string, int> name_to_rk;

static std :: vector <std :: string> rk_to_name;


int getNodeRank ()
{

  return rk;
}

int getNumberOfNodes ()
{

  return sz;
}

void collectiveCountOfRunners ( unsigned int* num_local_exec_runners, unsigned int* num_exec_runners )
{

  MPI_Allreduce( num_local_exec_runners, num_exec_runners, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD );
}

int getRankFromName (const std :: string & __name)
{

  return atoi (__name.c_str ());
}

void initNode (int * __argc, char * * * __argv)
{

  rk_to_name.clear ();
  name_to_rk.clear ();

  MPIThreadedEnv :: init ( __argc, __argv );
  //synchronizeNodes();

  MPI_Comm_rank (MPI_COMM_WORLD, & rk);   /* Who ? */
  MPI_Comm_size (MPI_COMM_WORLD, & sz);    /* How many ? */

  char names [sz] [MPI_MAX_PROCESSOR_NAME];
  int len;

  /* Processor names */
  MPI_Get_processor_name (names [0], & len);   /* Me */
  MPI_Allgather (names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD); /* Broadcast */

  for (int i = 0; i < sz; i ++)
    {
      rk_to_name.push_back (names [i]);
      name_to_rk [names [i]] = i;
    }
}
