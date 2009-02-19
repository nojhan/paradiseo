/*
* <moeoAllSolAllNeighborsExpl.h>
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

#ifndef _MOEOALLSOLALLNEIGHBORSEXPL_H
#define _MOEOALLSOLALLNEIGHBORSEXPL_H

#include <eo>
#include <moeo>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <moMoveIncrEval.h>

/**
 * TODO
 */
template < class Move >
class moeoAllSolAllNeighborsExpl : public moeoPopNeighborhoodExplorer < Move >
{
	typedef typename Move::EOType MOEOT;
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
public:
	
	moeoAllSolAllNeighborsExpl(
		moMoveInit < Move > & _moveInit,
		moNextMove < Move > & _nextMove,
		moMoveIncrEval < Move, ObjectiveVector > & _incrEval)
	: moveInit(_moveInit), nextMove(_nextMove), incrEval(_incrEval){}
	
	void operator()(eoPop < MOEOT > & _src, eoPop < MOEOT > & _dest)
	{
		Move move;
		ObjectiveVector objVec;
		for(unsigned int i=0; i<_src.size(); i++)
		{
			if (_src[i].flag() == 0)
			{
				moveInit(move, _src[i]);
				do
				{
					objVec = moveIncrEval(move, _src[i]);
					if (! comparator (objVec, _src[i].objectiveVector()))
					{
						if (objVec != _src[i].objectiveVector())
						{
							_dest.push_back(_src[i]);
							move(_dest.back());
							_dest.back().objectiveVector(objVec);
						}
					}
				}
				while(nextMove(move, _src[i]));
				_src[i].flag(1);
			}
		}		
	}
	
private:
	/** the move initializer */
	moMoveInit < Move > & moveInit;
	/** the neighborhood explorer */
	moNextMove < Move > & nextMove;
	/** the incremental evaluation */
	moMoveIncrEval < Move, ObjectiveVector > & incrEval;
	/** comparator */
	moeoParetoObjectiveVectorComparator<ObjectiveVector> comparator;

};

#endif /*_MOEOALLSOLALLNEIGHBORSEXPL_H_*/
