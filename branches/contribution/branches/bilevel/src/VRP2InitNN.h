#include <eoInit.h>
#include <VRP2.h>
#include <utils/eoRNG.h>

class VRP2InitNN: public eoInit<VRP2>{
	public:
		typedef std::multiset<DistMat::ty, DistMat::compaPair> setclas;
		VRP2InitNN( DistMat& _mat, eoRng& _rng, float _proba=1 ): mat(_mat), rng(_rng), proba(_proba) {}
		VRP2InitNN( DistMat& _mat, float _proba=1 ): mat(_mat),rng(eo::rng), proba(_proba) {}
		
		
		void operator()(VRP2 & _vrp){
			std::set<int> voituresPlacees;
			std::set<int> villesPlacees;
			std::vector<int> villesToAdd;
			bool dontstop;
			bool voiturePleine=true;

			int voituresParDepot=mat.numberOfVehicle()/mat.numberOfDepot();
			int voiture=-1;
			int ville=0;
			int lastPosWritten=_vrp.size();
			int depot=-1;
			int nextVille=-1;
			double charge=0;
			while (ville!=-1){
				if (voiturePleine){
					charge=0;
					villesToAdd.clear();
					villesToAdd.push_back(ville);
					villesPlacees.insert(ville);
					depot=-1;
					setclas depotclas=mat.nearDepot(ville);
					for (setclas::iterator it=depotclas.begin();it!=depotclas.end()&& depot==-1;it++){
						for (int i=0;i<voituresParDepot;i++){
							if(!voituresPlacees.count (voituresParDepot*((*it).first)+i)){
								depot=(*it).first;
								voiture=voituresParDepot*depot+i;
								break;
							}
						}
					}
					if (depot==-1) std::cout << "pas assez de voiture?"<<std::endl;
					voiturePleine=false;

				}
				dontstop=false;
				setclas clas=mat.getOrder(ville);
				do{
					for (setclas::iterator it=clas.begin();it!=clas.end();it++){
						if (!villesPlacees.count((*it).first) &&  mat.demand((*it).first)+charge <= mat.maxLoad())
							dontstop=true;
						if (!villesPlacees.count((*it).first) &&  mat.demand((*it).first)+charge <= mat.maxLoad()
								&& rng.flip(proba)){
							nextVille=(*it).first;
							break;
						}
					}
				}while(dontstop && nextVille==-1);
				if (nextVille!=-1) {
					villesToAdd.push_back(nextVille);
					villesPlacees.insert(nextVille);
					ville=nextVille;
					nextVille=-1;
				}else{
					int startToWriteIn=-1;
					if(voiture==0){
						for (unsigned int i=0; i< villesToAdd.size(); i++){
							_vrp[i]=villesToAdd[i];
						}
					}else{
						startToWriteIn=lastPosWritten-(villesToAdd.size()+1);
						_vrp[startToWriteIn]=voiture+mat.numberOfRetailer()-1;
						for (unsigned int i=0; i< villesToAdd.size(); i++){
							_vrp[startToWriteIn+1+i]=villesToAdd[i];
						}
						lastPosWritten=startToWriteIn;
						villesToAdd.clear();
					}
					voiturePleine=true;
					voituresPlacees.insert(voiture);
					ville=premiereVilleVide(_vrp,villesPlacees);
					villesToAdd.clear();
				}

			}
			if (voituresPlacees.size()<mat.numberOfVehicle()){
				for (unsigned int i=0;i<mat.numberOfVehicle()-1;i++){
					if (!voituresPlacees.count(i)){
						_vrp[lastPosWritten-- -1 ]=mat.numberOfRetailer()+i-1;
					}
				}
			}
			_vrp.invalidate();

		}




	private:
		DistMat &mat;
		eoRng &rng;
		float proba;




		//rend la premiere ville vide du tableau, avec une certaine proba d'en rendre une autre
		//assuré de rendre une ville vide q'il y en a une.
		int premiereVilleVide(VRP2 &vrp, std::set<int> &villesPlacees){
			int res=-1;
			bool dontstop=false;
			do{
				for (unsigned int i=0;i<mat.numberOfRetailer()&& res==-1; i++){
					if (!villesPlacees.count(i)) dontstop=true;
					if ( !villesPlacees.count(i) && rng.flip(proba))
						res=i;
				}
				if (res!=-1) return res;
			}while(dontstop);
			return res;
		}

};	
