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

template<class EOT>
paradiseo::smp::IslandModel<EOT>::IslandModel(AbstractTopology& _topo) :
    topo(_topo)
{ }

template<class EOT>
void paradiseo::smp::IslandModel<EOT>::add(AIsland<EOT>& _island)
{
    islands.push_back(&_island);
    islands.back()->setModel(this);
}

template<class EOT>
void paradiseo::smp::IslandModel<EOT>::operator()()
{
    std::vector<std::thread> threads(islands.size());
    
    // Construct topology according to the number of islands
    topo.construct(islands.size());
    
    // Create table
    table = createTable(topo, islands);
    
    // Lauching threads
    unsigned i = 0;
    for(auto it : islands)
    {
        threads[i] = std::thread(&AIsland<EOT>::operator(), it);
        i++;
    }

    unsigned workingThread = islands.size();
    while(workingThread > 0)
    {
        // Count working islands
        workingThread = islands.size();
        for(auto& it : islands)
            if(it->isStopped())
                workingThread--;
                
        // Check reception
        send();
        
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
        
    for(auto& thread : threads)
        thread.join();
    
    for(auto& message : sentMessages)
        message.join();   
}

template<class EOT>  
void paradiseo::smp::IslandModel<EOT>::update(eoPop<EOT> _data, AIsland<EOT>* _island)
{
    std::lock_guard<std::mutex> lock(m);
    //std::cout << "Pop reçue par le médiateur depuis " << _island << std::endl;
    //std::cout << _data << std::endl;
    listEmigrants.push(std::pair<eoPop<EOT>,AIsland<EOT>*>(_data, _island));
}

template<class EOT>      
void paradiseo::smp::IslandModel<EOT>::send(void)
{
    std::lock_guard<std::mutex> lock(m);
    while (!listEmigrants.empty())
    {
        std::cout << "Le mediateur va envoyer de " << listEmigrants.front().second  << " qui est " << table.getLeft()[listEmigrants.front().second] << std::endl;

        unsigned id = table.getLeft()[listEmigrants.front().second];
        std::vector<unsigned> neighbors = topo.getIdNeighbors(id);
        eoPop<EOT> migPop = listEmigrants.front().first;
        for (unsigned neighbor : neighbors)
        {
            std::cout << "On envoie à " << neighbor << std::endl;
            sentMessages.push_back(std::thread(&AIsland<EOT>::update, table.getRight()[neighbor], migPop));
		}    
        listEmigrants.pop();
    }
}

template<class EOT>     
Bimap<unsigned, AIsland<EOT>*> paradiseo::smp::IslandModel<EOT>::createTable(AbstractTopology& _topo, std::vector<AIsland<EOT>*>& _islands)
{
    Bimap<unsigned, AIsland<EOT>*> table;
    unsigned islandId = 0;
    for(auto it : islands)
    {
        table.add(islandId, it);
        islandId++;
    }
    
    return table;
}

