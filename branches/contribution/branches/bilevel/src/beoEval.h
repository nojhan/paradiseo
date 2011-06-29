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
//evaluator for bilevel problems using a differente operator for each level
#include <biVRP.h>
#include <BEO.h>
template <class BEOT> class beoEval: public eoEvalFunc<BEOT>{

	public:

		beoEval(eoEvalFunc<BEOT> &_evalL,eoEvalFunc<BEOT> &_evalU): evalL(_evalL),evalU(_evalU){}


		void operator()(BEOT &_beot){
			bool mode=_beot.getMode();
			if (_beot.invalid()){
				nbEval++;
				_beot.setMode(false);
				evalL(_beot);
				_beot.setMode(true);
				evalU(_beot);
			}
			if(mode){
				_beot.fitness(_beot.upper().fitness());
			}
			else{
				if (!_beot.lower().invalidFitness())
					_beot.fitness(_beot.lower().fitness());
			}
			_beot.setMode(mode);
		}

		int nbEval;


	private:
		eoEvalFunc<BEOT> &evalL;
		eoEvalFunc<BEOT> &evalU;

};
