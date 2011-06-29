#include <VRP2.h>
#include <op/VRP2Repair.h>
#include <biVRP.h>

class biVRPUpEval: public eoEvalFunc<biVRP>{
	public:
		biVRPUpEval(DistMat & _mat):mat(_mat){}


		void operator()(biVRP &_vrp){
			VRP2Eval evalup(mat);
			evalup(_vrp.upper());
			double neofit=-_vrp.upper().fitness();
			for (unsigned int i=0;i<mat.numberOfDepot();i++){
				for (unsigned int j=0;j<mat.numberOfPlant();j++){
					neofit+=_vrp.lower().sent(i,j,mat)*mat.cbOfPlant(j);
				}
			}
			_vrp.upper().fitness(-neofit);
			biVRP::ObjectiveVector obj;
			if(!_vrp.invalidObjectiveVector())
				obj=_vrp.objectiveVector();
			obj[0]=neofit;
			_vrp.setMode(true);
			_vrp.fitness(-neofit);
		}
	private:
		DistMat & mat;
};
