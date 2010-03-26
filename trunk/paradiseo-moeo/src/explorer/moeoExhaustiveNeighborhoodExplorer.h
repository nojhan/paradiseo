/*
* <moeoExhaustiveNeighborhoodExplorer.h>
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

#ifndef _MOEOEXHAUSTIVENEIGHBORHOODEXPLORER_H
#define _MOEOEXHAUSTIVENEIGHBORHOODEXPLORER_H

#include <eoPop.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moNeighborhood.h>
#include <explorer/moeoPopNeighborhoodExplorer.h>
#include <eval/moEval.h>

/**
 * Explorer which explore all the neighborhood
 */
template < class Neighborhood >
class moeoExhaustiveNeighborhoodExplorer : public moeoPopNeighborhoodExplorer < Neighborhood >
{
	/** Alias for the type */
    typedef typename Neighborhood::EOT MOEOT;
    typedef typename Neighborhood::Neighbor Neighbor;
    /** Alias for the objeciveVector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

public:

	/**
	 * Ctor
	 * @param _moveInit the move initializer
	 * @param _nextMove allow to do or not a move
	 * @param _incrEval a (generally) efficient evaluation fonction
	 */
    moeoExhaustiveNeighborhoodExplorer(
    	moNeighborhood<Neighbor>& _neighborhood,
    	moEval < Neighbor > & _eval):
    	neighborhood(_neighborhood), eval(_eval){}

    /**
     * functor to explore the neighborhood
     * @param _src the population to explore
     * @param _select contains index of individuals from the population to explore
     * @param _dest contains new generated individuals
     */
    void operator()(eoPop < MOEOT > & _src, std::vector < unsigned int> _select, eoPop < MOEOT > & _dest)
    {
        for(unsigned int i=0; i<_select.size(); i++)
        	explore(_src[_select[i]], _dest);
    }

private:

	/**
	 * explorer of one individual
	 * @param _src the individual to explore
	 * @param _dest contains new generated individuals
	 */
	void explore(MOEOT & _src , eoPop < MOEOT > & _dest)
	{
		if(neighborhood.hasNeighbor(_src)){
			neighborhood.init(_src, neighbor);
			_dest.push_back(_src);
			eval(_dest.back(),neighbor);
			neighbor.move(_dest.back());
			_dest.back().objectiveVector(neighbor.fitness());
			_dest.back().flag(0);
			while (neighborhood.cont(_src)){
				neighborhood.next(_src, neighbor);
				_dest.push_back(_src);
				eval(_dest.back(),neighbor);
				neighbor.move(_dest.back());
				_dest.back().objectiveVector(neighbor.fitness());
				_dest.back().flag(0);
			}
			_src.flag(1);
		}
	}

	/** Neighbor */
	Neighbor neighbor;
	/** Neighborhood */
	moNeighborhood<Neighbor>& neighborhood;
	/** ObjectiveVector */
    ObjectiveVector objVec;
    /** the incremental evaluation */
    moEval < Neighbor > & eval;

};

#endif /*_MOEOEXHAUSTIVENEIGHBORHOODEXPLORER_H_*/
