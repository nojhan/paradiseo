/*
* <scheduler.cpp>
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

#include <queue>
#include "synchron.h"
#include "../../core/messaging.h"
#include "node.h"
#include "tags.h"
#include "mess.h"



static SYNC syncRunners; /* Runners to be synchronized */

extern void wakeUpCommunicator();

extern RANK_ID getRankOfRunner (RUNNER_ID __key);

/* Initializing the list of runners to be synchronized */
void initSynchron ()
{

  syncRunners = SYNC();
}

/* packing a synchronization request from a service */
void packSynchronRequest ( const std :: vector <Cooperative *>& coops )
{

  /* Number of coops to synchronize */
  pack( (unsigned)( coops.size() ) );

  /* Coops to synchronize */
  for (unsigned i = 0; i < coops.size(); i ++)
    {
      pack( coops[ i ]->getOwner()->getDefinitionID() );
      pack( coops[ i ]->getKey() );
    }
}

/* Processing a synchronization request from a service */
void unpackSynchronRequest ()
{

  unsigned req_num_entries;
  unpack (req_num_entries);

  /* Creating a sync vector + adding the created entry */
  std::pair< SYNC_RUNNERS, unsigned > req_sync;

  /* Adding entries for each of the runners to be synchronized */
  SyncEntry req_entry;
  for (unsigned i = 0; i < req_num_entries; i ++)
    {

      unpack (req_entry.runner);
      unpack (req_entry.coop);

      req_sync.first.push_back (req_entry);
    }

  /* Looking for the sync vector */
  SYNC::iterator sync_it = syncRunners.find (req_sync);

  /* The vector does not exist - insert a new sync */
  if (sync_it == syncRunners.end ())
    {
      req_sync.second = 1;
      syncRunners.insert (req_sync);
    }
  else
    {

      /* The vector exists - updating the entry */
      std::pair< SYNC_RUNNERS, unsigned >& sync_req_entry = const_cast< std::pair< SYNC_RUNNERS, unsigned >& > (*sync_it);
      sync_req_entry.second ++;

      /* All the runners to be synchronized sent the SYNC_REQUEST signal */
      if (sync_req_entry.second == sync_req_entry.first.size())
        {

          /* Remove the entry */
          syncRunners.erase (sync_it);

          /* Send SYNCHRONIZED signals to all the coop objects */
          for (unsigned i = 0; i < req_sync.first.size(); i ++)
            {

              initMessage ();

              pack (req_sync.first [i].runner);
              pack (req_sync.first [i].coop);

              RANK_ID dest_rank = getRankOfRunner (req_sync.first [i].runner);
              sendMessage (dest_rank, SYNCHRONIZED_TAG);
            }
        }
    }
}
