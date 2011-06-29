#include <eoInit.h>
#include <PLA.h>
#include <utils/eoRNG.h>

class PLARandInit: public eoInit<PLA>{
	public:
		PLARandInit(DistMat &_mat, eoRng& _rng=eo::rng):rng(_rng),mat(_mat){}
		void operator()(PLA &_pla){
			_pla.init(mat);
			for (unsigned int i=0;i<_pla.size();i++){
				_pla[i]=rng.uniform();
			}
			_pla.repairPlants();
		}

	private:
		eoRng &rng;
		DistMat &mat;
};	
