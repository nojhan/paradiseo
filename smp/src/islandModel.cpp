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
void paradiseo::smp::IslandModel<EOT>::add(AIsland<EOT>& _island)
{
    static unsigned i = 0;
    islands[i] = &_island;
    islands[i]->setModel(this);
    i++;
}

template<class EOT>
void paradiseo::smp::IslandModel<EOT>::operator()()
{
    std::vector<Thread> threads(islands.size());

    unsigned i = 0;
    for(auto it : islands)
    {
        threads[i].start(&AIsland<EOT>::operator(), it.second);
        i++;
    }

    unsigned workingThread = islands.size();
    while(workingThread > 0)
    {
        workingThread = islands.size();
        for(auto& it : islands)
            if(it.second->isStopped())
                workingThread--;
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
        
    for(auto& thread : threads)
        thread.join();
        
}

template<class EOT>  
void paradiseo::smp::IslandModel<EOT>::update(eoPop<EOT> _data, AIsland<EOT>* _island)
{
    std::cout << "Pop reçue par le médiateur" << std::endl;
    std::lock_guard<std::mutex> lock(this->m);

    std::cout << _data << std::endl;
    listEmigrants.push(std::pair<eoPop<EOT>,AIsland<EOT>*>(_data, _island));
}
    


