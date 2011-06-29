#ifndef MOEOSTEADYCONTINUE_H_
#define MOEOSTEADYCONTINUE_H_
#include <set>
template <class MOEOT> class moeoSteadyContinue: public eoContinue<MOEOT>{


	public:
		moeoSteadyContinue(int _nbmin,int _nbmax,double _proportion=0.7):first(true),proportion(_proportion),nbmin(_nbmin),nbmax(_nbmax){}
		bool operator()(const eoPop<MOEOT> &_pop){
			static int count=0;
			if(first){
				current=0;
				steadystate=false;
				for (int x=0;x<_pop.size();x++){
					ol.insert(_pop[x].objectiveVector());
				}
				first=false;
				lastimpr=0;
				return true;
			}
			double countin=0;
			double countall=0;
			current++;

			for (int i=0;i<_pop.size();i++){
				if (ol.count(_pop[i].objectiveVector())!=0){
					countin++;
				}
				countall++;
			}
			ol.clear();
			for (int x=0;x<_pop.size();x++){
				ol.insert(_pop[x].objectiveVector());
			}
			if((countin/countall)>proportion){
				first=true;
				ol.clear();
				lastimpr=current;
			}
			return (current<nbmin) ||((current-lastimpr)<nbmax) ||  (countin/countall)<=proportion;
		}

	private:

		struct classcomp {
			bool operator() (const typename MOEOT::ObjectiveVector &o1, const typename MOEOT::ObjectiveVector &o2) const
			{
				return o1[0]<o2[0];
			}
		};
		bool first;
		std::set<typename MOEOT::ObjectiveVector,classcomp> ol;
		double proportion;
		int current;
		bool steadystate;
		int lastimpr;
		int nbmin;
		int nbmax;

};

#endif
