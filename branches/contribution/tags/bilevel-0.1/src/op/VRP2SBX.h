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
#include <VRP2.h>
#include <utils/eoRNG.h>
class VRP2SBX: public eoBinOp<VRP2>{

	public:
		VRP2SBX(float _proba=0.5, eoRng &_rnd=eo::rng):proba(_proba),rnd(_rnd){}
		bool operator()(VRP2 & vrp1, const VRP2 &vrp2){
			VRP2 res(vrp1);
			int link1=-1,link2=-1;
			int vehicle1;
			while (link1==-1 || vrp1.isVehicleAt(link1)){
				link1=rnd.random(vrp1.size());
			}
			while (link2==-1 || vrp2.isVehicleAt(link2)){
				link2=rnd.random(vrp2.size());
			}
			for (vehicle1=link1;vehicle1>=0;vehicle1--)
				if (vrp1.isVehicleAt(vehicle1)) 
					break;
			std::set<int> retailAfterLink2;
			for (unsigned int i =link2;i<vrp2.size() && ! vrp2.isVehicleAt(i);i++)
				retailAfterLink2.insert(vrp2[i]);
			int cur=0;
			int i=0;
			for (i=0;i<link1;i++){
				if (!retailAfterLink2.count(vrp1[i])){
					res[cur++]=vrp1[i];
				}
			}
			unsigned int cur2;
			for (cur2=link2;!vrp2.isVehicleAt(cur2)&& cur2 <vrp2.size();cur2++){			
				res[cur++]=vrp2[cur2];
			}
			for (unsigned int i=link1; i< vrp1.size() && cur<< vrp1.size(); i++){
				if (!retailAfterLink2.count(vrp1[i])){
					res[cur++]=vrp1[i];
				}
			}
			vrp1=res;
			vrp1.invalidate();
			return true;

		}
	private:
		float proba;
		eoRng &rnd;

};
