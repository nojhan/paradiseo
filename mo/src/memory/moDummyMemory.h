/*
<moDummyMemory.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moDummyMemory_h
#define _moDummyMemory_h

#include <memory/moMemory.h>

/**
 * Dummy memory to make an empty memory
 */
template< class Neighbor >
class moDummyMemory : virtual public moMemory<Neighbor> {

public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Init : NOTHIING TO DO
     */
    void init(EOT & /*_sol*/) {}

    /**
     * Add : NOTHIING TO DO
     */
    void add(EOT & /*_sol*/, Neighbor & /*_neighbor*/) {}

    /**
     * Update : NOTHIING TO DO
     */
    void update(EOT & /*_sol*/, Neighbor & /*_neighbor*/) {}

    /**
     * ClearMemory : NOTHIING TO DO
     */
    void clearMemory() {}

};


#endif
