/*
<MPI_IslandModel.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen
Eremey Valetov

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef SMP_MPI_ISLAND_MODEL_H_
#define SMP_MPI_ISLAND_MODEL_H_

#include <queue>
#include <list>
#include <algorithm>
#include <utility>
#include <future>
#include <thread>
#include <mpi.h>
#include <sstream>

#include <bimap.h>
#include <abstractIsland.h>
#include <topology/topology.h>

namespace paradiseo
{
namespace smp
{

/**
 * MPI-distributed island model.
 * Each MPI rank runs one island; migration is handled via MPI send/recv
 * using SerializableBase for population serialization.
 *
 * @see smp::Island, smp::MigPolicy
 */
template<class EOT>
class MPI_IslandModel
{
public:
    /**
     * @param _topo topology defining island connectivity
     * @param _pollIntervalMs polling interval in milliseconds (default 1000)
     */
    MPI_IslandModel(AbstractTopology& _topo, int _pollIntervalMs = 1000);

    void add(AIsland<EOT>& _island);
    void operator()();
    bool update(eoPop<EOT> _data, AIsland<EOT>* _island);
    void setTopology(AbstractTopology& _topo);
    bool isRunning() const;

protected:
    void send(void);
    void initModel(void);
    Bimap<unsigned, AIsland<EOT>*> createTable();

    // Non-blocking MPI send: buffer must stay alive until send completes
    struct PendingSend {
        MPI_Request request;
        std::string buffer;
    };

    void completePendingSends();

    std::queue<std::pair<eoPop<EOT>,AIsland<EOT>*>> listEmigrants;
    Bimap<unsigned, AIsland<EOT>*> table;
    std::vector<std::pair<AIsland<EOT>*, bool>> islands;
    AbstractTopology& topo;
    std::vector<std::shared_future<bool>> sentMessages;
    std::list<PendingSend> pendingSends;
    std::mutex m;
    int mpi_rank, num_mpi_ranks;
    int pollIntervalMs;
    std::atomic<bool> running;
};

#include <MPI_IslandModel.cpp>

}
}

#endif
