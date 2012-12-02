/*
<islandModel.h>
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

#ifndef ISLAND_MODEL_H_
#define ISLAND_MODEL_H_

#include <queue>
#include <algorithm>
#include <utility>
#include <future>
#include <thread>
#include <bimap.h>
#include <abstractIsland.h>
#include <topology/topology.h>

namespace paradiseo
{
namespace smp
{

/** IslandModel

The IslandModel object is an island container that provides mecanism in order to perform a
island model pattern according to a topology.

@see smp::Island, smp::MigPolicy
*/



template<class EOT>
class IslandModel
{
public:
    IslandModel(AbstractTopology& topo);

    /**
     * Add an island to the model.
     * @param _island Island to add.
     */
    void add(AIsland<EOT>& _island);

    /**
     * Launch the island model by starting islands on their population.
     */
    void operator()();
    
    /**
     * Update the island model by adding population to send in the emigrants list.
     */
    void update(eoPop<EOT> _data, AIsland<EOT>* _island);
    
    /**
     * Change topology
     * @param _topo New topology.
     */
    void setTopology(AbstractTopology& _topo);
    
    bool isRunning() const;
    
protected:
    
    /**
     * Send population to islands
     */
    void send(void);
    
    Bimap<unsigned, AIsland<EOT>*> createTable();

    std::queue<std::pair<eoPop<EOT>,AIsland<EOT>*>> listEmigrants;
    Bimap<unsigned, AIsland<EOT>*> table;
    std::vector<std::pair<AIsland<EOT>*, bool>> islands;
    AbstractTopology& topo;
    std::vector<std::thread> sentMessages;
    std::mutex m;
    std::atomic<bool> running;
};

#include <islandModel.cpp>

}

}

#endif
