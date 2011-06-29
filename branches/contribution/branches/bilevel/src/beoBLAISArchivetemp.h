/*
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2010
*
* Legillon Francois
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
//Coevolutionary bilevel method using repeated algorithms (CoBRA)
//algorithm to solve bilevel problems
#ifndef BEOBLAISARCHIVE3_H_
#define BEOBLAISARCHIVE3_H_
#include <BEO.h>
#include <eoAlgo.h>
#include <eoOp.h>
#include <eoEvalFunc.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoContinue.h>
#include <eoMerge.h>
#include <beoCoevoPop.h>
#include <eoPopEvalFunc.h>
#include <eoReplacement.h>
#include <beoSelectUp.h>
#include <beoSelectLow.h>
#include <eoStochasticUniversalSelect.h>
#include <eoPerf2Worth.h>
#include <eoRandomSelect.h>
#include <iostream>
#include <algorithm>
#include <archive/moeoArchive.h>
#include <archive/moeoUnboundedArchive.h>
template <class BEOT> 
class beoBLAISArchivetemp: public eoAlgo<BEOT> {
	public:
		typedef typename BEOT::U U;
		typedef typename BEOT::L L;
		typedef typename BEOT::Fitness Fitness;
		typedef typename U::Fitness FitnessU;
		typedef typename L::Fitness FitnessL;

		beoBLAISArchivetemp(	
				eoAlgo<BEOT>&, 
				eoAlgo<BEOT>&,
				eoEvalFunc<BEOT>&, 
				beoCoevoPop<BEOT>&,
				eoContinue<BEOT>&,
				eoContinue<BEOT>&,
				eoSelectOne<BEOT>&,
				eoSelectOne<BEOT>&,
				eoReplacement<BEOT>&,
				unsigned int,
				eoRng &_rng
			       ); 

		void operator()(eoPop<BEOT> &_pop);
	private:
		struct s_fitcomp{
			bool operator()(const BEOT& b1, const BEOT& b2){
				return b1.upper().fitness()>b2.upper().fitness();
			}
		}fitcomp;
		eoAlgo<BEOT>& algoU; 
		eoAlgo<BEOT>& algoL;
		eoEvalFunc<BEOT>& eval;
		beoCoevoPop<BEOT>& coevo;
		eoContinue<BEOT>& contL;
		eoContinue<BEOT>& contU;
		eoSelectOne<BEOT>& selectU;
		eoSelectOne<BEOT>& selectL;
		eoReplacement<BEOT> &merge;
		unsigned int nbFromArc;
		eoRng &rng;

		void setPopMode(eoPop<BEOT> &_pop, bool mode){
			for (unsigned int i=0;i< _pop.size(); i++){
				_pop[i].setMode(mode);
				_pop[i].invalidate();
				eval(_pop[i]);
			}
		}

		void archiveBest(eoPop<BEOT> &_pop,eoPop<BEOT> &_archive, bool level){
			_archive.sort();
			bool empty=_archive.empty();
			for (unsigned int i=0;i<_pop.size();i++){
				typename BEOT::U up=_pop[i].upper();
				typename BEOT::L low=_pop[i].lower();
				if(empty) {
					_archive.push_back(_pop[i]);
					empty=false;
				}
				else{
					bool ok=true;
					for (unsigned int j=0;j<_archive.size();j++){
						if ((level && _archive[j].lower()==low)||(!level && _archive[j].upper()==up)){
							ok=false;
							break;
						}
					}
					if(ok)
						_archive.push_back(_pop[i]);
				}

			}
		}

};	
#endif
