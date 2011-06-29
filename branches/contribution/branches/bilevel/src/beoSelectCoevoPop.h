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
//beoSelectCoevoPop selects a part of two population and then co-evolute them together
template <class BEOT> 
class beoSelectCoevoPop: public beoCoevoPop<BEOT>{
	public:
		beoSelectCoevoPop(
				eoSelect<BEOT> &_select1,
				eoSelect<BEOT> &_select2, 
				beoCoevoOp<BEOT> &_op):
			select1(_select1), 
			select2(_select2),
			op(_op),
			poper(op)
	{}
		
		beoSelectCoevoPop(
				eoSelect<BEOT> &_select1,
				beoCoevoOp<BEOT> &_op):
			select1(_select1), 
			select2(_select1),
			op(_op),
			poper(op)
	{}
		void operator()(eoPop<BEOT> &_pop1,eoPop<BEOT> &_pop2){
			eoPop<BEOT> res1,res2;
			select1(_pop1,res1);
			select2(_pop2,res2);
			poper(res1,res2);
			_pop1=res1;
			_pop2=res2;
		}
	private:
		eoSelect<BEOT> &select1;
		eoSelect<BEOT> &select2;
		beoCoevoOp<BEOT> &op;
		beoCoevoAllPop<BEOT> poper;

};
