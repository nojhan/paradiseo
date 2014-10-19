/*
<moCombinedContinuator.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

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

#ifndef _moCombinedContinuator_h
#define _moCombinedContinuator_h

#include "moContinuator.h"
#include "../neighborhood/moNeighborhood.h"
#include <vector>

/**
 * Combined several continuators.
 * Continue until one of the continuators returns false
 */
template< class Neighbor >
class moCombinedContinuator : public moContinuator<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    /**
     * Constructor (moCheckpoint must have at least one continuator)
     * @param _cont a continuator
     */
    moCombinedContinuator(moContinuator<Neighbor>& _cont) {
        continuators.push_back(&_cont);
    }

    /**
     * add a continuator to the combined continuator
     * @param _cont a continuator
     */
    void add(moContinuator<Neighbor>& _cont) {
        continuators.push_back(&_cont);
    }

    /**
     * init all continuators
     * @param _solution a solution
     */
    virtual void init(EOT & _solution) {
        for (unsigned int i = 0; i < continuators.size(); ++i)
            continuators[i]->init(_solution);
    }

    /**
     *@param _solution a solution
     *@return true all the continuators are true
     */
    virtual bool operator()(EOT & _solution) {
        bool bContinue = true;

        // some data may be update in each continuator.
        // So, all continuators are tested
        for (unsigned int i = 0; i < continuators.size(); ++i)
            if ( !(*continuators[i])(_solution) )
                bContinue = false;

        return bContinue;
    }

private:
    /** continuators vector */
    std::vector< moContinuator<Neighbor>* > continuators;

};
#endif
