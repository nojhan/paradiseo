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
//Quadratic operator adaptator, to cross 2 bilevel solutions along their lower level
#include <BEO.h>
#include <eoOp.h>

template <class BEOT> 
class beoLowQuad : public eoQuadOp<BEOT>{
	public:
		typedef typename BEOT::L L;
		beoLowQuad(eoQuadOp<L> &_quadl):quadl(_quadl){}
		bool operator()(BEOT &l1, BEOT &l2){
			if(quadl(l1.lower(),l2.lower())){
				l1.invalidate();
				l1.upper().invalidate();
				l1.lower().invalidate();
				l2.invalidate();
				l2.upper().invalidate();
				l2.lower().invalidate();
				return true;
			}
			return false;
		}
	private:
		eoQuadOp<typename BEOT::L> &quadl;


};
