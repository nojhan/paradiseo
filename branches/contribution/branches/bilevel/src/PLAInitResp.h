#ifndef PLAINIRESP_H_
#define PLAINIRESP_H_
#include <eoInit.h>
#include <PLA.h>
#include <utils/eoRNG.h>

class PLAInitResp: public eoInit<biVRP>{
	public:
		PLAInitResp(DistMat &_mat, eoRng& _rng=eo::rng):rng(_rng),mat(_mat){}
		void operator()(biVRP &_vrp){
			PLA &_pla=_vrp.lower();
			VRP2 &vrp=_vrp.upper();
			_pla.init(mat);
			std::vector<double> usines;
			usines.resize(mat.numberOfPlant(),0);
			for (unsigned int dep=0;dep<mat.numberOfDepot();dep++){
				double demand=vrp.depotDemand(mat,dep);
				double sent=0;
				for (unsigned int usi=0;usi<usines.size() && demand>sent;usi++){
					if (usines[usi]<1){
						double eps=0.000001;
						double resttaux=1-usines[usi];
						double tosend=demand-sent;
						if (tosend<eps)break;

						if (resttaux*mat.availability(usi)<(tosend)){
							sent+=resttaux*mat.availability(usi);
							usines[usi]=1;
							_pla[dep*mat.numberOfPlant()+usi]=resttaux;
						}else{
							_pla[dep*mat.numberOfPlant()+usi]=tosend/mat.availability(usi);
							usines[usi]+=tosend/mat.availability(usi);
							sent=demand;
						}
					}
				}
			}

			
			for (unsigned int i=0;i<_pla.size();i++){
				_pla[i]=rng.uniform();
			}
			_pla.repairPlants();
		}

	private:
		eoRng &rng;
		DistMat &mat;
};
#endif
