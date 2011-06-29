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
//beoCoevoAllPop applies a coevolution operator to all couples of individuals of pop1xpop2
//then copy the results in both population
template <class BEOT> class beoCoevoAllPop: public beoCoevoPop<BEOT>{
	public:
		beoCoevoAllPop(beoCoevoOp<BEOT> &_op):op(_op){}

		void operator()(eoPop<BEOT> &_pop1, eoPop<BEOT> &_pop2){
			unsigned int size1=_pop1.size();
			unsigned int size2=_pop2.size();

			for (unsigned int i=0;i<size1;i++){
				for (unsigned int j=0;j<size2;j++){
					BEOT neo1=_pop1[i];
					BEOT neo2=_pop2[j];
					op(neo1,neo2);
					_pop1.push_back(neo1);
					_pop1.push_back(neo2);
					_pop2.push_back(neo1);
					_pop1.push_back(neo2);
				}
			}

		}

	private:
		beoCoevoOp<BEOT> &op;
};
