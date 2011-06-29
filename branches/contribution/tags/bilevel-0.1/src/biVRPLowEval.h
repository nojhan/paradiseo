#include <VRP2.h>
#include <op/VRP2Repair.h>
#include <biVRP.h>
#include <PLA.h>

class biVRPLowEval: public eoEvalFunc<biVRP>{
	public:
		biVRPLowEval(DistMat & _mat,double _lambda=1000.):mat(_mat),lambda(_lambda){}


		void operator()(biVRP &_vrp){
			double res=0;
			_vrp.lower().repairPlants();
			for (unsigned int i=0;i<mat.numberOfDepot();i++){
				VRP2 vrp(_vrp.upper());
				double demand=vrp.depotDemand(mat,i);
				double sent=0;
				for (unsigned int j=0;j<mat.numberOfPlant();j++){
					res+=_vrp.lower().sent(i,j,mat)*(mat.ccOfPlant(j)+mat.plantDepotDistance(i,j)/10000);
					sent+=_vrp.lower().sent(i,j,mat);
				}
				if (sent<demand) {
					res+=(demand-sent)*lambda;
				}else{
				}
			}
			_vrp.lower().fitness(-res);
			_vrp.setMode(false);
			_vrp.fitness(-res);
		}
	private:
		DistMat & mat;
		double lambda;
};
