/*
<policy.h>
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

#ifndef MIG_POLICY_H_
#define MIG_POLICY_H_

#include <set>

#include <eo>
#include <migPolicyElement.h>
#include <island.h>
#include <abstractIsland.h>

namespace paradiseo
{
namespace smp
{
// Forward declaration
template <class EOT>
class AIsland;

template <class EOT>
class Policy : public eoContinue<EOT>, public std::vector<PolicyElement<EOT>>
{
public:
    bool operator()(const eoPop<EOT>& _pop)
    {
        std::cout << "On regarde la politique de migration" << std::endl;
        for(PolicyElement<EOT>& elem : *this)
        {
            std::cout << ".";
            if(!elem(_pop))
            {
                std::cout << "On lance l'emmigration" << std::endl;
                notifyIsland(elem.getSelect());
            }
        }     
        return true; // Always return true because it never stops the algorithm
    }
    
    void addObserver(AIsland<EOT>* _observer)
    {
        observers.insert(_observer);
    }
 
    void removeObserver(AIsland<EOT>* _observer)
    {
        observers.erase(_observer);
    }
   
protected:

    void notifyIsland(eoSelect<EOT>& _select) const
    {
        std::cout << "On notifie les iles" << std::endl;
        for (AIsland<EOT>* it : observers)
            it->send(_select);
    }

    std::set<AIsland<EOT>*> observers;
};

}

}

#endif
