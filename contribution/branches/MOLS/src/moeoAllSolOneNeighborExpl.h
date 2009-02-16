/*
* <moeoAllSolOneNeighborExpl.h>
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

#ifndef _MOEOALLSOLONENEIGHBOREXPL_H
#define _MOEOALLSOLONENEIGHBOREXPL_H

#include <eo>
#include <moeo>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <queue>

/**
 * TODO
 */
template < class Move >
class moeoAllSolOneNeighborExpl : public moeoPopNeighborhoodExplorer < Move >
{
	typedef typename Move::EOType MOEOT;
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
public:
	
	moeoAllSolOneNeighborExpl(
		moMoveInit < Move > & _moveInit,
		moNextMove < Move > & _nextMove,
		eoEvalFunc < MOEOT > & _eval
	): moveInit(_moveInit), nextMove(_nextMove), eval(_eval){}
	
	void operator()(eoPop < MOEOT > & _src, eoPop < MOEOT > & _dest){
//Move move;
		MOEOT * sol;
		ObjectiveVector objVec;
		int id=0; 
		
		for(unsigned int i=0; i<_src.size(); i++){
			//solution without move
			if(_src[i].flag() == 0 ){
				//No move are available -> create a new Move
				if(availableMove.empty()){
					//create a new move
					Move newMove;
					//add it to moveVector
					moveVector.push_back(newMove);
					//get the moveVector size
					id = moveVector.size();
					//add a flag to _src
					_src[i].flag(-id);
					//Init the move
					moveInit(moveVector[id-1], _src[i]);
					//copy the solution in destination population
					_dest.push_back(_src[i]);
					//stock the new solution
					sol = & _dest.back();
					//apply the move on it
					moveVector[id-1](*sol);
					//invalidate it
					sol->invalidate();
					//evaluate ir
					eval(*sol);
					//set it as not yet visisted (flag=0)
					sol->flag(0);
					//If it the last move set solution as visited (flag >0) and set the move as available
					if(!nextMove(moveVector[id-1], _src[i])){
						_src[i].flag(1);
						availableMove.push(id-1);
					}
				}
				//A move is available -> get it
				else{
					//get the id of an available move
					id = availableMove.back();
					//remove it from available move
					availableMove.pop();
					//add a flag to _src
					_src[i].flag(-1 * (id+1));
					//Init the move
					moveInit(moveVector[id], _src[i]);
					//copy the solution in destination population
					_dest.push_back(_src[i]);
					//stock the new solution
					sol = & _dest.back();
					//apply the move on it
					moveVector[id](*sol);
					//invalidate it
					sol->invalidate();
					//evaluate ir
					eval(*sol);
					//set it as not yet visisted (flag=0)
					sol->flag(0);
					if(!nextMove(moveVector[id], _src[i])){
						_src[i].flag(1);
						availableMove.push(id);
					}
				}
			}
			//solution which have already a move -> do next move
			else if(_src[i].flag() < 0){
				id= (_src[i].flag() + 1) * -1;
				_dest.push_back(_src[i]);
				sol = & _dest.back();
				//objVec = moveIncrEval(move, *sol);
				moveVector[id](*sol);
				sol->invalidate();
				eval(*sol);
				//sol->objectiveVector(objVec);
				//if (comparator(sol, _src[i]))		
				if(!nextMove(moveVector[id], _src[i])){
					_src[i].flag(1);
					availableMove.push(id);
				}								
			}
		}
	}
	
private:
	/** queue of available move */
	std::queue < unsigned int > availableMove;
	/** the move vector*/
	std::vector < Move > moveVector;
	/** the move initializer */
	moMoveInit < Move > & moveInit;
	/** the neighborhood explorer */
	moNextMove < Move > & nextMove;
	/** the incremental evaluation */
	eoEvalFunc < MOEOT > & eval;
	
};

#endif /*_MOEOALLSOLONENEIGHBOREXPL_H_*/
