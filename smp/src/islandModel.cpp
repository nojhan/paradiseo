/*
<islandModel.cpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Alexandre Quemy, Thibault Lasnier - INSA Rouen

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
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

#include <functional>
#include <algorithm>

template<class EOT>
paradiseo::smp::IslandModel<EOT>::IslandModel(AbstractTopology& _topo) :
    topo(_topo),
    running(false)
{ }

template<class EOT>
void paradiseo::smp::IslandModel<EOT>::add(AIsland<EOT>& _island)
{
    islands.push_back(std::pair<AIsland<EOT>*, bool>(&_island, false));
    islands.back().first->setModel(this);
}

template<class EOT>
void paradiseo::smp::IslandModel<EOT>::operator()()
{
    running = true;

    // INIT PART
    // Create topology, table and initialize islands
    initModel();
    
    std::vector<std::thread> threads(islands.size());

    // Launching threads
    unsigned i = 0;
    for(auto& it : islands)
    {
        it.first->setRunning();
        threads[i] = std::thread(&AIsland<EOT>::operator(), it.first);
        i++;
    }

    // Lambda function in order to know the number of working islands
    std::function<int()> workingIslands = [this]() -> int
    {
        return (int)std::count_if(std::begin(islands), std::end(islands),
            [](std::pair<AIsland<EOT>*, bool>& i) -> bool
            { return i.second; } );
    };
    
    // SCHEDULING PART
    while(workingIslands() > 0)
    {
        // Count working islands
        for(auto& it : islands)
        {
            // If an island is stopped we need to isolate its node in the topology
            if(it.second && it.first->isStopped())
            {
                it.second = false;
                topo.isolateNode(table.getLeft()[it.first]);
            }
        }        
        // Check sending
        send();
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
    
    // ENDING PART
    // Wait the end of algorithms
    for(auto& thread : threads)
        thread.join();
    
   
    // Wait the end of messages sending
    for(auto& message : sentMessages)
        message.wait();
        
    // Clear the sentMessages container
    sentMessages.clear();
        
    // Force last integration
    while(!listEmigrants.empty())
        send();
    i = 0;
    for(auto& it : islands)
    {
        threads[i] = std::thread(&AIsland<EOT>::receive, it.first);
        i++;
    }

    // Wait the end of the last integration
    for(auto& thread : threads)
        thread.join();
        
    running = false;
    std::cout << "hhhhhhh" << listEmigrants.size() << std::endl;
}

template<class EOT>  
bool paradiseo::smp::IslandModel<EOT>::update(eoPop<EOT> _data, AIsland<EOT>* _island)
{
    std::lock_guard<std::mutex> lock(m);
    listEmigrants.push(std::pair<eoPop<EOT>,AIsland<EOT>*>(_data, _island));
    
    return true;
}

template<class EOT>  
void paradiseo::smp::IslandModel<EOT>::setTopology(AbstractTopology& _topo)
{
    // If we change topo, we need to protect it
    std::lock_guard<std::mutex> lock(m);
    topo = _topo;
    // If we change when the algorithm is running, we need to recontruct the topo
    if(running)
    {
        topo.construct(islands.size());
        // If we change the topology during the algorithm, we need to isolate stopped islands
        for(auto it : islands)
            if(!it.second)
                topo.isolateNode(table.getLeft()[it.first]);
    }
}

template<class EOT>      
void paradiseo::smp::IslandModel<EOT>::send(void)
{
    std::lock_guard<std::mutex> lock(m);
    if (!listEmigrants.empty())
    {
        // Get the neighbors
        unsigned idFrom = table.getLeft()[listEmigrants.front().second];
        std::vector<unsigned> neighbors = topo.getIdNeighbors(idFrom);
        
        // Send elements to neighbors
        eoPop<EOT> migPop = std::move(listEmigrants.front().first);
        sentMessages.erase(std::remove_if(sentMessages.begin(), sentMessages.end(), 
            [&](std::shared_future<bool>& i) -> bool
            { return i.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready; }
            ), 
            sentMessages.end());
        for (unsigned idTo : neighbors)
            sentMessages.push_back(std::async(std::launch::async, &AIsland<EOT>::update, table.getRight()[idTo], std::move(migPop))); 
             
        listEmigrants.pop();
    }

}

template<class EOT>     
bool paradiseo::smp::IslandModel<EOT>::isRunning() const
{
    return (bool)running;
}

template<class EOT> 
void paradiseo::smp::IslandModel<EOT>::initModel(void)
{
    // Preparing islands
    for(auto& it : islands)
        it.second = true; // Indicate islands are active
    
    // Construct topology according to the number of islands
    topo.construct(islands.size());
    
    // Create table
    table = createTable();
}

template<class EOT>     
Bimap<unsigned, AIsland<EOT>*> paradiseo::smp::IslandModel<EOT>::createTable()
{
    Bimap<unsigned, AIsland<EOT>*> table;
    unsigned islandId = 0;
    for(auto it : islands)
    {
        table.add(islandId, it.first);
        islandId++;
    }
    
    return table;
}
