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

#ifndef SMP_POLICY_ELEMENT_H_
#define SMP_POLICY_ELEMENT_H_

#include <eo>

namespace paradiseo
{
namespace smp
{
/** PolicyElement: PolicyElement is an element of a migration policy.

The policy element is a pair made of a selection method and a criterion to apply the selection.

*/

template <class EOT>
class PolicyElement : public eoContinue<EOT>
{
public :
    /**
     * Constructor
     * @param _selection How to select elements for migration
     * @param _criterion When notifying the island
     */
    PolicyElement(eoSelect<EOT>& _selection, eoContinue<EOT>& _criterion);
    
    /**
     * Check is the criterion is reach
     * @param _pop Population which is checked by the criteria.
     * @return false if the criterion is reached.
     */
    bool operator()(const eoPop<EOT>& _pop);
    
    /**
     * Add criteria for the same selection method.
     * @param _criterion New criterion.
     */
    void addCriterion(eoContinue<EOT>& _criterion);
    
     /**
     * Access to the selection method.
     * @return Reference to the selection method.
     */
    eoSelect<EOT>& getSelect();
    
protected :
    eoSelect<EOT>& selection;
    eoContinue<EOT>& criterion;
};

#include <policyElement.cpp>

}

}

#endif
