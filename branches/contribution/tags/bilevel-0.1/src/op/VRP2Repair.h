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
#ifndef VRP2REPAIR_H_
#define VRP2REPAIR_H_
#include <VRP2.h>
#include <utils/eoRNG.h>
#include <cmath>
class VRP2Repair : public eoMonOp<VRP2>  {

	public:
		VRP2Repair(DistMat &_mat, eoRng& _rng, float _bias = 0.5 ):rng(_rng),gen(_bias,_rng),mat(_mat){}
		VRP2Repair(DistMat &_mat, float _bias = 0.5):rng(eo::rng),gen(_bias,rng),mat(_mat){}

		bool operator()(VRP2 &_vrp){
			VRP2 res(_vrp);
			double chargeMax=mat.maxLoad();
			std::list<int> listRes;
			std::list<std::pair <std::list<int>::iterator, double > > itesLibres;
			bool first=true;
			unsigned int currentVehicle=0;			
			unsigned int i;
			std::vector<int> malplaces;

			while(first ||currentVehicle<_vrp.size()){
				double charge;
				if(first)charge=_vrp.chargeOfVehicleAt(mat,-1);
				else charge=_vrp.chargeOfVehicleAt(mat,currentVehicle);
				bool OK=charge<=chargeMax;
				if (OK) {
					for(i=(first?0:currentVehicle+1);i<_vrp.size() && !_vrp.isVehicleAt(i);i++){
						listRes.push_back(_vrp[i]);
					}
					if (i<_vrp.size()){
						std::list<int>::iterator it=listRes.insert(listRes.end(),_vrp[i]);
						if (_vrp.isVehicleAt(i)){
							std::make_pair(it,charge);
							itesLibres.push_back(std::make_pair(it,charge));
						}
					}
				}
				else{
					charge=0;
					for (i=(first?0:currentVehicle+1); i<_vrp.size() && !_vrp.isVehicleAt(i); i++){
						if (charge+mat.demand(_vrp[i])<=chargeMax ){
							charge+=mat.demand(_vrp[i]);
							listRes.push_back(_vrp[i]);
						}else{
							malplaces.push_back(i);
						}
					}
					listRes.push_back(_vrp[i]);
				}
				currentVehicle=i;
				first=false;
			}
			for (unsigned int j=0;j<malplaces.size();j++){
				bool done=false;
				std::list<std::pair< std::list<int>::iterator, double> >::iterator it;
				i=malplaces[j];
				for(it=itesLibres.begin();it!=itesLibres.end()&&!done;it++){
					if((*it).second+mat.demand(_vrp[i])<=chargeMax){
						done=true;
						listRes.insert((*it).first,_vrp[i]);
						(*it).second+=mat.demand(_vrp[i]);
					}
				}
				if(!done) {
					listRes.push_back(_vrp[i]);
				}
			}
			int cur=0;
			for (std::list<int>::iterator it=listRes.begin();it!=listRes.end();it++){
				_vrp[cur++]=*it;
			}
			_vrp.invalidate();
			return true;
		}


	private:
		eoRng &rng;
		eoBooleanGenerator gen;
		DistMat &mat;


};
#endif
