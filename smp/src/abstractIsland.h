/*
<abstractIsland.h>
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

#ifndef ABS_ISLAND_H_
#define ABS_ISLAND_H_

#include <atomic>

#include <eo>
#include <migPolicy.h>

namespace paradiseo
{
namespace smp
{

// Forward declaration
template<class EOT>
class IslandModel;

/** AbstractIsland: An abstract island.

The abstract island is used to manipulate island pointers wihout the knowledge of the algorithm.

@see smp::Island
*/

template<class EOT>
class AIsland
{
public:

    virtual void operator()() = 0;
    
    /**
     * Check if there is population to receive
     */
    virtual void setModel(IslandModel<EOT>* _model) = 0;
    

    
    /**
     * Check if there is population to receive or to emigrate
     */
    virtual void check(void) = 0;
    
    /**
     * Update the island  by adding population to send in the imigrants list.
     */
    virtual void update(eoPop<EOT> _data) = 0;
    
    /**
     * Check if the algorithm is stopped.
     * @return true if stopped
     */
    virtual bool isStopped(void) const = 0;

protected:
    std::mutex m;    
};

}

}

#endif
