#include <VRP2.h>
#include <op/VRP2Repair.h>
#include <biVRP.h>
#include <PLA.h>
#include <mbiVRPDistMat.h>
class mbiVRPLowEval: public eoEvalFunc<biVRP>{
	public:
		mbiVRPLowEval(mbiVRPDistMat& _mat,double _lambda=1000.):mat(_mat),lambda(_lambda){}


		void operator()(biVRP &_vrp){
			biVRP::ObjectiveVector vecres;
			if (_vrp.invalidObjectiveVector()){
				_vrp.objectiveVector(vecres);
			}
			PLAObjectiveVector res;
			res[0]=0;
			res[1]=0;
			_vrp.lower().repairPlants();
			for (unsigned int i=0;i<mat.numberOfDepot();i++){
				VRP2 vrp(_vrp.upper());
				double demand=vrp.depotDemand(mat,i);
				double sent=0;
				for (unsigned int j=0;j<mat.numberOfPlant();j++){
					res[0]+=_vrp.lower().sent(i,j,mat)*(mat.ccOfPlant(j)+mat.plantDepotDistance(i,j)/10000);
					res[1]+=_vrp.lower().sent(i,j,mat)*(mat.cdOfPlant(j)+mat.plantDepotDistance(i,j)/10000);
					sent+=_vrp.lower().sent(i,j,mat);
				}
				if (sent<demand) {
					res[0]+=(demand-sent)*lambda;
					res[1]+=(demand-sent)*lambda;
				}
			}
			biVRP::ObjectiveVector obj;
			obj.lowset(res);
			obj.upset(_vrp.objectiveVector().up());
			_vrp.objectiveVector(obj);
			_vrp.lower().fitness(0);
		}
	private:
		mbiVRPDistMat & mat;
		double lambda;
};
