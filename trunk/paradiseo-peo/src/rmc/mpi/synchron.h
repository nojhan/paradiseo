/*
* <synchron.h>
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

#ifndef __synchron_h
#define __synchron_h

#include <set>
#include <vector>
#include <utility>

#include "../../core/runner.h"
#include "../../core/cooperative.h"

struct SyncEntry
  {

    RUNNER_ID runner;
    COOP_ID coop;
  };

struct SyncCompare
  {

    bool operator()( const std::pair< std::vector< SyncEntry >, unsigned >& A, const std::pair< std::vector< SyncEntry >, unsigned >& B )
    {

      const std::vector< SyncEntry >& syncA = A.first;
      const std::vector< SyncEntry >& syncB = B.first;

      if ( syncA.size() == syncB.size() )
        {
          std::vector< SyncEntry >::const_iterator itA = syncA.begin();
          std::vector< SyncEntry >::const_iterator itB = syncB.begin();

          while ( (*itA).runner < (*itB).runner && itA != syncA.end() )
            {
              itA++;
              itB++;
            }

          return itA == syncA.end();
        }

      return syncA.size() < syncB.size();
    }
  };

typedef std::vector< SyncEntry > SYNC_RUNNERS;
typedef std::set< std::pair< SYNC_RUNNERS, unsigned >, SyncCompare > SYNC;

/* Initializing the list of runners to be synchronized */
extern void initSynchron ();

/* packing a synchronization request from a service */
extern void packSynchronRequest ( const std :: vector <Cooperative *>& coops );

/* Processing a synchronization request from a service */
extern void unpackSynchronRequest ();

#endif
