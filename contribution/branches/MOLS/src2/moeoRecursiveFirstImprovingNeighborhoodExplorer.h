/*
* <moeoRecursiveFirstImprovingNeighborhoodExplorer.h>
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

#ifndef _MOEORECURSIVEFIRSTIMPROVINGNEIGHBORHOODEXPLORER_H
#define _MOEORECURSIVEFIRSTIMPROVINGNEIGHBORHOODEXPLORER_H

#include <moeoSubNeighborhoodExplorer.h>

/**
 * Explorer which explore a part of the neighborhood
 */
template < class Move >
class moeoRecursiveFirstImprovingNeighborhoodExplorer : public moeoSubNeighborhoodExplorer < Move >
{
	/** Alias for the type */
    typedef typename Move::EOType MOEOT;
    /** Alias for the objeciveVector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    using moeoSubNeighborhoodExplorer<Move>::move;
    using moeoSubNeighborhoodExplorer<Move>::objVec;
    using moeoSubNeighborhoodExplorer<Move>::number;

public:

	/**
	 * Ctor
	 * @param _moveInit the move initializer
	 * @param _nextMove allow to do or not a move
	 * @param _incrEval a (generally) efficient evaluation fonction
	 * @param _number the number of neighbor to explore
	 */
    moeoRecursiveFirstImprovingNeighborhoodExplorer(
        moMoveInit < Move > & _moveInit,
        moNextMove < Move > & _nextMove,
        moMoveIncrEval < Move, ObjectiveVector > & _incrEval)
            : moeoSubNeighborhoodExplorer< Move >(_moveInit, _nextMove, _incrEval, 0){}

private:

	/**
	 * explorer of one individual
	 * @param _src the individual to explore
	 * @param _dest contains new generated individuals
	 */
	void explore(MOEOT & _src, eoPop < MOEOT > & _dest)
	{
		moveInit(move, _src);
		do
		{
			objVec = incrEval(move, _src);
			if(!comparator(objVec, _src.objectiveVector())){
				_dest.push_back(_src);
				move(_dest.back());
				_dest.back().objectiveVector(objVec);
				_dest.back().flag(0);
			}
		}
		while (nextMove(move, _src) && (!comparator( _src.objectiveVector(),objVec)));
		if(comparator( _src.objectiveVector(),objVec))
			explore(_dest.back(), _dest);
		else
			_src.flag(1);
	}

	/** Objective Vector Pareto Comparator */
	moeoParetoObjectiveVectorComparator<ObjectiveVector> comparator;
};

#endif /*_MOEORECURSIVEFIRSTIMPROVINGNEIGHBORHOODEXPLORER_H_*/
