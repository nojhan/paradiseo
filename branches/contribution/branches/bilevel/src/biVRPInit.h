#include <biVRP.h>
#include <biVRPDistMat.h>
#include <PLAInitResp.h>
class biVRPInit : public eoInit<biVRP>{
	public:
		biVRPInit(DistMat& _mat):mat(_mat), initpla(mat), initNN(mat),initVRP(initNN),initbeo(initVRP,initpla) {}
		biVRPInit(DistMat& _mat,eoInit<VRP2> &_initVRP):mat(_mat), initpla(mat), initNN(mat), initVRP(_initVRP),initbeo(initVRP,initpla) {}
		void operator()(biVRP &_vrp){
			_vrp.upper().init(mat);
			_vrp.lower().init(mat);
			initbeo(_vrp);
		}
	private:
		DistMat &mat;
		PLAInitResp initpla;
		VRP2InitNN initNN;
		eoInit<VRP2> &initVRP;
		beoInit<biVRP> initbeo;
};
