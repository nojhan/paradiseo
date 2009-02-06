/*
* <moeoOneSolOneNeighborExpl.h>
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

#ifndef _MOEOONESOLONENEIGHBOREXPL_H
#define _MOEOONESOLONENEIGHBOREXPL_H

#include <eo>
#include <moeo>
#include <moMove.h>
#include <moMoveInit.h>
#include <moNextMove.h>
#include <move/moeoMoveIncrEval.h>

/**
 * TODO
 */
template < class Move >
class moeoOneSolOneNeighborExpl : public moeoPopNeighborhoodExplorer < Move >
{
	typedef typename Move::EOType MOEOT;
	typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	
public:
	
	moeoOneSolOneNeighborExpl(
		moMoveInit < Move > & _moveInit,
		moNextMove < Move > & _nextMove,
		eoEvalFunc < MOEOT > & _eval
	):start(false), moveInit(_moveInit), nextMove(_nextMove), eval(_eval){}
	
	void operator()(eoPop < MOEOT > & _src, eoPop < MOEOT > & _dest){
		//Move move;
		MOEOT * sol;
		ObjectiveVector objVec;
		
			
		unsigned int i = 0;
		while(_src[i].flag() != 0)
			i++;
		if(!start){
			moveInit(move, _src[i]);
			_dest.push_back(_src[i]);
			sol = & _dest.back();
			move(*sol);
			sol->invalidate();
			eval(*sol);
			if(nextMove(move, _src[i]))
				start=true;
			else{
				start=false;
				_src[i].flag(1);
			}
		}
		else{
			_dest.push_back(_src[i]);
			sol = & _dest.back();
			//objVec = moveIncrEval(move, *sol);
			move(*sol);
			sol->invalidate();
			eval(*sol);
			//sol->objectiveVector(objVec);
			//if (comparator(sol, _src[i]))		
			if(!nextMove(move, _src[i])){
				start=false;
				_src[i].flag(1);
			}				
		}	
	}
	
private:
	Move move;
	bool start;
	/** the move initializer */
	moMoveInit < Move > & moveInit;
	/** the neighborhood explorer */
	moNextMove < Move > & nextMove;
	/** the incremental evaluation */
	eoEvalFunc < MOEOT > & eval;
	
};

#endif /*_MOEOONESOLONENEIGHBOREXPL_H_*/
