#include <eoPerf2Worth.h>
class biVRPPerf2Worth: public eoPerf2WorthCached<biVRP,biVRP::Fitness>{
	using eoPerf2WorthCached<biVRP,biVRP::Fitness>::value;
	virtual void calculate_worths(const eoPop<biVRP> &_pop){
		value().resize(_pop.size());
		for (unsigned int i=0;i<_pop.size();i++){
			value()[i]=1/(-_pop[i].fitness());
		}
	}
};
