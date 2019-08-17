/*
<moLocalSearchInit.h>
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

#ifndef _moLocalSearchInit_h
#define _moLocalSearchInit_h

#include <paradiseo/eo/eoInit.h>
#include "../algo/moLocalSearch.h"

/**
 * Initialization of the solution with a local search
 */
template< class Neighbor >
class moLocalSearchInit : public eoInit<typename Neighbor::EOT> {
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _init initialization of the solution before the local search
     * @param _ls the local search to apply to the solution
     */
    moLocalSearchInit(eoInit<EOT>& _init, moLocalSearch<Neighbor>& _ls) : init(_init), ls(_ls) {
    }

    /**
     * Apply the local search on the solution
     * @param _solution to perturb
     */
    void operator()(EOT& _solution) {
        init(_solution);
        ls(_solution);
    }

private:
    eoInit<EOT>& init;
    moLocalSearch<Neighbor> & ls;
};

#endif
