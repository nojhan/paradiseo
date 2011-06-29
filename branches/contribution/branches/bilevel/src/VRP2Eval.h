#include <VRP2.h>
#include <op/VRP2Repair.h>

class VRP2Eval: public eoEvalFunc<VRP2> {
	public:
		VRP2Eval (DistMat &_mat): mat(_mat){}

		void operator()(VRP2 &_vrp){
			VRP2Repair repair(mat);
			repair(_vrp);
			double neo_fit=0;
			double vehicle=0;
			bool lastWasVehicle=true;
			int lastvehicle=0;
			bool firstVehicle=true;
			if (!_vrp.isVehicleAt(0)){
				neo_fit=mat.depotDistance(0,_vrp[0]);
			}

			for (unsigned int i=0;i<_vrp.size();i++){
				if(_vrp.isVehicleAt(i)){
					if(!lastWasVehicle){
						if (i==0) std::cout<<"tinn"<<std::endl;
						vehicle+=mat.depotDistance(firstVehicle?0:_vrp.depotOfVehicleAt(lastvehicle),_vrp[i-1]);
					}
					firstVehicle=false;
					lastvehicle=i;
					neo_fit+=vehicle;
					vehicle=0;
					lastWasVehicle=true;
				}else{
					if (lastWasVehicle && lastvehicle){
						vehicle+=mat.depotDistance(_vrp.depotOfVehicleAt(i-1),_vrp[i]);
						lastWasVehicle=false;
					}else{
						if(i!=0){
							vehicle+=mat.distance(_vrp[i-1],_vrp[i]);

						}
						lastWasVehicle=false;
					}
				}

			}
			neo_fit+=vehicle;
			_vrp.fitness(-neo_fit);
		}
	private:
		DistMat &mat;

};
