/*
   <moeoVNS.h>
   Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
   (C) OPAC Team, LIFL, 2002-2008

   Sébastien Cahon, Jean-Charles Boisson (Jean-Charles.Boisson@lifl.fr)
   François Legillon

   This software is governed by the CeCILL license under French law and
   abiding by the rules of distribution of free software.  You can  use,
   modify and/ or redistribute the software under the terms of the CeCILL
   license as circulated by CEA, CNRS and INRIA at the following URL
   "http://www.cecill.info".

   As a counterpart to the access to the source code and  rights to copy,
   modify and redistribute granted by the license, users are provided only
   with a limited warranty  and the software's author,  the holder of the
   economic rights,  and the successive licensors  have only  limited liability.

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

#ifndef _moeoVNS_h
#define _moeoVNS_h

#include "algo/moeoAlgo.h"
#include "../../eo/eoEvalFunc.h"
#include "algo/moeoSolAlgo.h"
#include <moHCMoveLoopExpl.h>
#include "../fitness/moeoSingleObjectivization.h"
#include "explorer/moeoHCMoveLoopExpl.h"
//! Variable Neighbors Search (VNS)
/*!
  Class which describes the algorithm for a Variable Neighbors Search.
  Adapts the moVNS for a multi-objective problem using a moeoSingleObjectivization.
  M is for Move
 */

template < class MOEOT >
class moeoVNS:public moeoSolAlgo < MOEOT >
{

	public:
//		typedef typename M::EOType MOEOT;
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename MOEOT::Fitness Fitness;
		//! Generic constructor
		/*!
		  Generic constructor using a moExpl

		  \param _explorer Vector of Neighborhoods.
		  \param _full_evaluation The singleObjectivization containing a full eval.
		 */
		moeoVNS(moExpl< MOEOT> & _explorer, moeoSingleObjectivization < MOEOT> & _full_evaluation): algo(_explorer,_full_evaluation) {}
		
		//! Function which launches the VNS
		/*!
		  The VNS has to improve a current solution.

		  \param _solution a current solution to improve.
		  \return true.
		 */
		bool operator()(MOEOT &_solution){
			return algo(_solution);
		}

	private:
		//! the actual algo
		moVNS<MOEOT> algo;

};
#endif
