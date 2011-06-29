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
//
//Coevolutionary bilevel method using repeated algorithms (CoBRA)
//algorithm to solve bilevel problems
#include <beoBLAISArchivetemp.h>
#include <utils/eoRNG.h>

template <class BEOT> 
beoBLAISArchivetemp<BEOT>::beoBLAISArchivetemp(
		eoAlgo<BEOT>& _au, 
		eoAlgo<BEOT>& _al,
		eoEvalFunc< BEOT>&_eval, 
		beoCoevoPop< BEOT> &_coevo, 
		eoContinue< BEOT> &_contL, 
		eoContinue< BEOT> &_contU, 
		eoSelectOne<BEOT>& _selectU,
		eoSelectOne<BEOT>& _selectL,
		eoReplacement<BEOT> &_merge,
		unsigned int _nbFromArc=20,
		eoRng &_rng=eo::rng
		):
	algoU(_au),
	algoL(_al),
	eval(_eval),
	coevo(_coevo), 
	contL(_contL), 
	contU(_contU), 
	selectU(_selectU),
	selectL(_selectL),
	merge(_merge),
	nbFromArc(_nbFromArc),
	rng(_rng)
{}



template <class BEOT> 
void beoBLAISArchivetemp<BEOT>::operator()(eoPop<BEOT> &_pop){
	unsigned taille=_pop.size();
	for (unsigned int i=0;i<_pop.size();i++){
		eval(_pop[i]);
	}

	eoPop <BEOT> archiveUp;
	moeoUnboundedArchive <BEOT> archiveLow;
	eoPop <BEOT> neopopl;
	eoPop <BEOT> neopopu;
	eoPop<BEOT> popu=_pop;
	eoPop<BEOT> popl=_pop;

	setPopMode(popl,false);
	setPopMode(popu,true);
	algoU(popu);
	setPopMode(popu,true);
	algoL(popl);
	bool continu=contU(popu);
	bool continl=contL(popl);

	while(continu && continl){
		//on eval les pop, on copie
		setPopMode(popl,false);
		setPopMode(popu,true);
		neopopu=popu;
		neopopl=popl;

		//on pompe l'archive
		setPopMode(archiveLow,false);
		setPopMode(archiveUp,true);
		setPopMode(popu,true);
		if (archiveUp.size()>1) 
			selectU.setup(archiveUp);
		if (archiveLow.size()>1)
			selectL.setup(archiveLow);
		unsigned int boucle=nbFromArc;
		if (boucle> archiveUp.size()) boucle=archiveUp.size();
		if (boucle> archiveLow.size()) boucle=archiveLow.size();
		for (unsigned int i=0;i<boucle;i++){
			BEOT up=selectU(archiveUp);
			BEOT low=selectL(archiveLow);
			if (find(neopopu.begin(),neopopu.end(),up)==neopopu.end())neopopu.push_back(up);
			if (find(neopopl.begin(),neopopl.end(),low)==neopopl.end())neopopl.push_back(low);
		}

		//on coevolue
		coevo(neopopu,neopopl);
		setPopMode(neopopl,false);
		setPopMode(neopopu,true);

		//on optimise
		algoU(neopopu);
		algoL(neopopl);
		setPopMode(neopopl,false);
		setPopMode(neopopu,true);

		//on archive et on verifie la taille des archives
		archiveBest(neopopu,archiveUp,true);
		archiveLow(neopopl);
		if(archiveUp.size()> _pop.size()){
			archiveUp.sort();
			archiveUp.resize(_pop.size());	
		}
		if(archiveLow.size()> _pop.size()){
			std::sort(archiveLow.begin(),archiveLow.end(),fitcomp);
			archiveLow.resize(_pop.size());	
		}

		neopopu.sort();
		std::unique(neopopu.begin(),neopopu.end());
		eoPop<BEOT>popuclean;
		for (unsigned int i=0;i<neopopu.size();i++){
			if (std::find(popuclean.begin(),popuclean.end(),neopopu[i])==popuclean.end()&&
					std::find(popu.begin(),popu.end(),neopopu[i])==popu.end()){
				popuclean.push_back(neopopu[i]);
			}
		}
		neopopl.sort();
		std::unique(neopopl.begin(),neopopl.end());
		eoPop<BEOT>poplclean;
		for (unsigned int i=0;i<neopopl.size();i++){
			if (std::find(poplclean.begin(),poplclean.end(),neopopl[i])==poplclean.end()&&
					std::find(popl.begin(),popl.end(),neopopl[i])==popl.end())
				poplclean.push_back(neopopl[i]);
		}

		//on merge
		setPopMode(popl,false);
		setPopMode(popu,true);
		setPopMode(popuclean,true);
		setPopMode(poplclean,false);
		merge(popu,popuclean);
		merge(popl,poplclean);
		continu=contU(popu);
		continl=contL(popl);


	}
	//on copie les popu et popl, ainsi que les archives dans pop, 
	//puis on tronque
	_pop.resize(archiveUp.size()+archiveLow.size());
	std::copy(archiveUp.begin(),archiveUp.end(),_pop.begin());
	std::copy(archiveLow.begin(),archiveLow.end(),_pop.begin()+archiveUp.size());
	setPopMode(_pop,true);
	_pop.sort();
	_pop.resize(taille);

	for(int i=0;i<popu.size();i++){
		std::cout<<"popul"<<popu[i].lower()<<std::endl;
		std::cout<<"popuu"<<popu[i].upper()<<std::endl;
		std::cout<<"popufit"<<popu[i].upper().fitness()<<" "<<popu[i].objectiveVector().low()[0]<<" "<<popu[i].objectiveVector().low()[1]<<std::endl;
	}
	for(int i=0;i<popl.size();i++){
		std::cout<<"popll"<<popl[i].lower()<<std::endl;
		std::cout<<"poplu"<<popl[i].upper()<<std::endl;
		std::cout<<"poplfit"<<popl[i].upper().fitness()<<" "<<popl[i].objectiveVector().low()[0]<<" "<<popl[i].objectiveVector().low()[1]<<std::endl;
	}
	for(int i=0;i<archiveUp.size();i++){
		std::cout<<"arcul"<<archiveUp[i].lower()<<std::endl;
		std::cout<<"arcuu"<<archiveUp[i].upper()<<std::endl;
		std::cout<<"arcufit"<<archiveUp[i].upper().fitness()<<" "<<archiveUp[i].objectiveVector().low()[0]<<" "<<archiveUp[i].objectiveVector().low()[1]<<std::endl;
	}
	for(int i=0;i<archiveLow.size();i++){
		std::cout<<"arcll"<<archiveLow[i].lower()<<std::endl;
		std::cout<<"arclu"<<archiveLow[i].upper()<<std::endl;
		std::cout<<"arclfit"<<archiveLow[i].upper().fitness()<<" "<<archiveLow[i].objectiveVector().low()[0]<<" "<<archiveLow[i].objectiveVector().low()[1]<<std::endl;
	}
}
