/*
<policyElement.h>
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

#ifndef POLICY_ELEM_H_
#define POLICY_ELEM_H_

#include <eo>

namespace paradiseo
{
namespace smp
{

template <class EOT>
class PolicyElement : public eoContinue<EOT>
{
public :
    PolicyElement(eoSelect<EOT>& _selection, eoContinue<EOT>& _criteria) :
        selection(_selection),
        criteria(_criteria)
    {
        
    }
    
    bool operator()(const eoPop<EOT>& _pop)
    {
        // DEBUG
        static int i = 0;
        std::cout << i << std::endl;   
        i++;
        // END DEBUG
        
        return criteria(_pop);
    }
    
    void addCriteria(eoContinue<EOT>& _criteria)
    {
        criteria.add(_criteria);
    }
    
    eoSelect<EOT>& getSelect() 
    {
        return selection;
    }
    
protected :
    eoSelect<EOT>& selection;
    eoContinue<EOT>& criteria;
};

}

}

#endif
