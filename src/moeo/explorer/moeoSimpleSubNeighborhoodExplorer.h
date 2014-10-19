/*
* <moeoSubNeighborhoodExplorer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jérémie Humeau
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------

#ifndef _MOEOSIMPLESUBNEIGHBORHOODEXPLORER_H
#define _MOEOSIMPLESUBNEIGHBORHOODEXPLORER_H

#include "moeoSubNeighborhoodExplorer.h"

/**
 * Explorer which explore a part of the neighborhood
 */
template < class Neighbor >
class moeoSimpleSubNeighborhoodExplorer : public moeoSubNeighborhoodExplorer < Neighbor >
{
	/** Alias for the type */
    typedef typename Neighbor::EOT MOEOT;
    /** Alias for the objeciveVector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    using moeoSubNeighborhoodExplorer<Neighbor>::neighborhood;
    using moeoSubNeighborhoodExplorer<Neighbor>::neighbor;
    using moeoSubNeighborhoodExplorer<Neighbor>::number;

public:

	/**
	 * Constructor
	 * @param _neighborhood a neighborhood
	 * @param _number number of neighbor to explore
	 * @param _eval a neighbor evaluation function
	 */
    moeoSimpleSubNeighborhoodExplorer(
			moNeighborhood<Neighbor>& _neighborhood,
	        unsigned int _number,
        	moEval < Neighbor > & _eval)
            : moeoSubNeighborhoodExplorer<Neighbor>(_neighborhood, _number), eval(_eval){}

private:

	/**
	 * explorer of one individual
	 * @param _src the individual to explore
	 * @param _dest contains new generated individuals
	 */
	void explore(MOEOT & _src, eoPop < MOEOT > & _dest)
	{
		unsigned int tmp=number;
		//if the neighborhood is not empty
		if(neighborhood.hasNeighbor(_src) && tmp>0){
			//init the neighborhood
			neighborhood.init(_src, neighbor);
			//eval the neighbor
			cycle(_src, _dest);
			tmp--;
			//repeat all instructions for each neighbor in the neighborhood until a best neighbor is found
			while (neighborhood.cont(_src) && tmp>0){
				neighborhood.next(_src, neighbor);
				cycle(_src, _dest);
				tmp--;
			}
			//if all neighbors are been visited, fix the source flag to 1 (visited solution)
			if(!neighborhood.cont(_src))
				_src.flag(1);
		}
	}

	/**
	 * subfunction of explore
	 * @param _src the individual to explore
	 * @param _dest contains new generated individuals
	 */
	void cycle(MOEOT & _src, eoPop < MOEOT > & _dest){
		eval(_src, neighbor);
		//copy the solution (_src) at the end of the destination (_dest)
		_dest.push_back(_src);
		//move the copy
		neighbor.move(_dest.back());
		//affect objective vector to the copy
		_dest.back().objectiveVector(neighbor.fitness());
		//fix its flag to 0 (unvisited solution)
		_dest.back().flag(0);
	}

	/** Incremental evaluation of a neighbor */
	moEval < Neighbor > & eval;

};

#endif /*_MOEOSIMPLESUBNEIGHBORHOODEXPLORER_H_*/
