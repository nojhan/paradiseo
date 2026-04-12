/*
<MPI_IslandModel.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen
Eremey Valetov

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <functional>
#include <algorithm>
#include <mpi.h>
#include <SerializableBase.h>
#include <serial/Parser.h>
#include <utils/eoLogger.h>

template<class EOT>
MPI_IslandModel<EOT>::MPI_IslandModel(AbstractTopology& _topo, int _pollIntervalMs)
    : topo(_topo), pollIntervalMs(_pollIntervalMs), running(false)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
}

template<class EOT>
void paradiseo::smp::MPI_IslandModel<EOT>::add(AIsland<EOT>& _island)
{
    islands.push_back(std::pair<AIsland<EOT>*, bool>(&_island, false));
    islands.back().first->setModel(this);
}

template<class EOT>
void MPI_IslandModel<EOT>::operator()() {
    running = true;
    initModel();

    std::thread islandThread;
    if (mpi_rank < static_cast<int>(islands.size())) {
        auto& it = islands[mpi_rank];
        it.first->setRunning();
        islandThread = std::thread(&AIsland<EOT>::operator(), it.first);
    }

    int localIslandRunning = 1;
    int globalIslandRunning = 0;

    do {
        localIslandRunning = (mpi_rank < static_cast<int>(islands.size()) &&
                              !islands[mpi_rank].first->isStopped()) ? 1 : 0;

        send();

        MPI_Allreduce(&localIslandRunning, &globalIslandRunning, 1,
                       MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (globalIslandRunning > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
    } while (globalIslandRunning > 0);

    if (islandThread.joinable())
        islandThread.join();

    for (auto& message : sentMessages)
        message.wait();
    sentMessages.clear();

    // Discard remaining outgoing emigrants — all islands have stopped,
    // so migrated individuals would never be processed by recipients.
    // Using send() here risks MPI deadlock when multiple ranks each
    // have pending emigrants to send but no matching receives posted.
    {
        std::lock_guard<std::mutex> lock(m);
        while (!listEmigrants.empty()) listEmigrants.pop();
    }

    std::thread lastIntegrationThread;
    if (mpi_rank < static_cast<int>(islands.size())) {
        lastIntegrationThread = std::thread([&]() {
            islands[mpi_rank].first->receive();
        });
    }

    if (lastIntegrationThread.joinable())
        lastIntegrationThread.join();

    // Wait for any async island.update() tasks launched during the drain phase.
    // Without this, islands may be deleted while tasks still reference them.
    for (auto& message : sentMessages)
        message.wait();
    sentMessages.clear();

    // Cancel remaining non-blocking sends. After the polling loop and drain,
    // any unsent data targets stopped islands — safe to discard.
    for (auto& ps : pendingSends) {
        MPI_Cancel(&ps.request);
        MPI_Wait(&ps.request, MPI_STATUS_IGNORE);
    }
    pendingSends.clear();

    running = false;
}

template<class EOT>
bool paradiseo::smp::MPI_IslandModel<EOT>::update(eoPop<EOT> _data, AIsland<EOT>* _island)
{
    std::lock_guard<std::mutex> lock(m);
    listEmigrants.push(std::pair<eoPop<EOT>,AIsland<EOT>*>(_data, _island));
    return true;
}

template<class EOT>
void paradiseo::smp::MPI_IslandModel<EOT>::setTopology(AbstractTopology& _topo)
{
    std::lock_guard<std::mutex> lock(m);
    topo = _topo;
    if (running) {
        topo.construct(islands.size());
        for (auto it : islands)
            if (!it.second)
                topo.isolateNode(table.getLeft()[it.first]);
    }
}

template<class EOT>
void paradiseo::smp::MPI_IslandModel<EOT>::send(void) {
    // Receive first, then send — prevents accumulation of unprocessed messages.
    if (!islands.empty() && mpi_rank < static_cast<int>(islands.size())) {
        unsigned myId = mpi_rank;
        std::vector<unsigned> neighbors = topo.getIdNeighbors(myId);

        for (unsigned idFromNeighbor : neighbors) {
            int tag = idFromNeighbor * 1000 + mpi_rank;
            int flag = 0;
            MPI_Status mpiStatus;
            MPI_Iprobe(idFromNeighbor, tag, MPI_COMM_WORLD, &flag, &mpiStatus);

            if (flag) {
                // Single-message receive: MPI_Iprobe confirmed the complete
                // message is available, so MPI_Recv returns immediately.
                // This avoids the two-message protocol (size + data) in
                // comm.recv(), which can deadlock when the size message
                // arrives before the data message is progressed by MPI.
                int count = 0;
                MPI_Get_count(&mpiStatus, MPI_CHAR, &count);
                std::string serialized(count, '\0');
                MPI_Recv(&serialized[0], count, MPI_CHAR, idFromNeighbor, tag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                eoserial::Object* obj = eoserial::Parser::parse(serialized);
                SerializableBase<eoPop<EOT>> receivedSerializablePop;
                receivedSerializablePop.unpack(obj);
                delete obj;

                eoPop<EOT> receivedPop = receivedSerializablePop;
                eo::log << eo::debug << "MPI_IslandModel: rank " << mpi_rank
                        << " received " << receivedPop.size() << " migrant(s) from island "
                        << idFromNeighbor << " (tag=" << tag << ")" << std::endl;

                sentMessages.push_back(std::async(std::launch::async,
                    &AIsland<EOT>::update,
                    table.getRight()[myId],
                    std::move(receivedPop)));
            }
        }
    }

    // Clean up completed async tasks
    sentMessages.erase(std::remove_if(sentMessages.begin(), sentMessages.end(),
        [](std::shared_future<bool>& i) -> bool
        { return i.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready; }
        ),
        sentMessages.end());

    // Then send outgoing emigrants
    eoPop<EOT> migPop;
    unsigned idFrom = 0;
    bool hasMigrant = false;
    {
        std::lock_guard<std::mutex> lock(m);
        if (!listEmigrants.empty()) {
            idFrom = table.getLeft()[listEmigrants.front().second];
            migPop = std::move(listEmigrants.front().first);
            listEmigrants.pop();
            hasMigrant = true;
        }
    }
    if (hasMigrant) {
        std::vector<unsigned> neighbors = topo.getIdNeighbors(idFrom);

        // Serialize once for all neighbors (same protocol as comm.send for Persistent)
        SerializableBase<eoPop<EOT>> serializablePop(migPop);
        eoserial::Object* obj = serializablePop.pack();
        std::stringstream ss;
        obj->print(ss);
        delete obj;
        std::string serialized = ss.str();
        int size = static_cast<int>(serialized.size()) + 1;

        for (unsigned idTo : neighbors) {
            int tag = idFrom * 1000 + idTo;
            eo::log << eo::debug << "MPI_IslandModel: rank " << mpi_rank
                    << " sending " << migPop.size() << " migrant(s) from island "
                    << idFrom << " to island " << idTo << " (tag=" << tag << ")" << std::endl;

            // Single non-blocking send. The receiver uses MPI_Get_count
            // from MPI_Iprobe to determine the size, eliminating the
            // separate blocking size message that could deadlock.
            pendingSends.emplace_back();
            PendingSend& ps = pendingSends.back();
            ps.buffer = serialized;
            MPI_Isend(ps.buffer.data(), size, MPI_CHAR, idTo, tag,
                      MPI_COMM_WORLD, &ps.request);
        }
    }

    completePendingSends();
}

template<class EOT>
void paradiseo::smp::MPI_IslandModel<EOT>::completePendingSends() {
    auto it = pendingSends.begin();
    while (it != pendingSends.end()) {
        int completed = 0;
        MPI_Test(&it->request, &completed, MPI_STATUS_IGNORE);
        if (completed)
            it = pendingSends.erase(it);
        else
            ++it;
    }
}

template<class EOT>
bool paradiseo::smp::MPI_IslandModel<EOT>::isRunning() const
{
    return (bool)running;
}

template<class EOT>
void paradiseo::smp::MPI_IslandModel<EOT>::initModel(void)
{
    if (num_mpi_ranks > static_cast<int>(islands.size()) + 1) {
        eo::log << eo::errors << "MPI_IslandModel: number of MPI ranks ("
                << num_mpi_ranks << ") exceeds number of islands + 1 ("
                << islands.size() + 1 << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (auto& it : islands)
        it.second = true;

    topo.construct(islands.size());
    table = createTable();
}

template<class EOT>
Bimap<unsigned, AIsland<EOT>*> paradiseo::smp::MPI_IslandModel<EOT>::createTable()
{
    Bimap<unsigned, AIsland<EOT>*> table;
    unsigned islandId = 0;
    for (auto it : islands) {
        table.add(islandId, it.first);
        islandId++;
    }
    return table;
}
