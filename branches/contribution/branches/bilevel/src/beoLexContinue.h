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
//continuator stopping when no improvment was made during a parametrized number of generation
#include <eoContinue.h>
template<class BEOT>
class beoLexContinue: public eoSteadyFitContinue<BEOT>{
	public:
		typedef typename BEOT::Fitness Fitness;
		using eoSteadyFitContinue<BEOT>::steadyState;
		using eoSteadyFitContinue<BEOT>::thisGeneration;
		using eoSteadyFitContinue<BEOT>::lastImprovement;
		using eoSteadyFitContinue<BEOT>::repSteadyGenerations;
		using eoSteadyFitContinue<BEOT>::repMinGenerations;

		beoLexContinue( unsigned long _minGens, unsigned long _steadyGens,bool _onlyup=true):eoSteadyFitContinue<BEOT>(_minGens,_steadyGens),onlyup(_onlyup){}

		beoLexContinue( unsigned long _minGens, unsigned long _steadyGen,unsigned long& _currentGen,bool _onlyup=true):
			eoSteadyFitContinue<BEOT>(_minGens,_steadyGen,_currentGen),onlyup(_onlyup)
	{}
		virtual bool operator()(const eoPop<BEOT> &pop){
			thisGeneration++;

			Fitness bestCurrentUpFitness = pop[0].upper().fitness();
			Fitness bestCurrentLowFitness = onlyup?0:pop[0].lower().fitness();
			for (unsigned int i=1;i<pop.size();i++){
				if (pop[i].upper().fitness()>bestCurrentUpFitness 
						|| 
						(pop[i].upper().fitness()==bestCurrentUpFitness 
						 &&(onlyup|| pop[i].lower().fitness()>bestCurrentLowFitness))){
					bestCurrentUpFitness=pop[i].upper().fitness();
					bestCurrentLowFitness=onlyup?0:pop[i].lower().fitness();
				}
			}


			if (steadyState){     
				if (bestCurrentUpFitness > bestUpSoFar || (bestCurrentUpFitness==bestUpSoFar && bestCurrentLowFitness>bestLowSoFar)){
					bestUpSoFar = bestCurrentUpFitness;
					bestLowSoFar = bestCurrentLowFitness;
					lastImprovement = thisGeneration;
				}else{
					if(thisGeneration-lastImprovement>repSteadyGenerations){
						return false;
					}
				} 
			} else {
				if (thisGeneration > repMinGenerations) { 
					steadyState = true;
					bestUpSoFar = bestCurrentUpFitness;
					bestLowSoFar = bestCurrentLowFitness;
					lastImprovement = thisGeneration;
				}
			}
			return true;
		}

		Fitness bestUpSoFar;
		Fitness bestLowSoFar;
		bool onlyup;



};

