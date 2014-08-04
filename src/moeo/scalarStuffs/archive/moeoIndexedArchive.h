#ifndef MOEOINDEXEDARCHIVE_H_
#define MOEOINDEXEDARCHIVE_H_

#include "../../../eo/eoPop.h"
#include "../../archive/moeoArchive.h" 
#include "moeoArchiveIndex.h"

/**
 * Archive used for 2 dimension vectors which remove pareto dominated values
 * Use an moeoArchiveIndex
 */
template < class MOEOT >
class moeoIndexedArchive : public moeoArchive < MOEOT >
{

	public:


		using eoPop < MOEOT > :: size;
		using eoPop < MOEOT > :: operator[];
		using eoPop < MOEOT > :: pop_back;

		/**
		 * The type of an objective vector for a solution
		 */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;

		/**
		 * Default ctor.
		 * The moeoObjectiveVectorComparator used to compare solutions is based on Pareto dominance
		 */
		moeoIndexedArchive(moeoArchiveIndex<MOEOT>& _index) : index(_index) {}


		/**
		 * Updates the archive with a given individual _moeo
		 * @param _moeo the given individual
		 */
		bool operator()(const MOEOT & _moeo){
			std::pair<bool,std::vector<typename moeoArchiveIndex<MOEOT>::modif> > res=index(_moeo);
			if (!(res.first)){
				return false;
			}
			else{
				for (unsigned int i=0;i<res.second.size();i++){
					apply_modif(res.second[i]);
				}
				this->push_back(_moeo);
				return true;
			}
		}

		/**
		 * Updates the archive with a given population _pop
		 * @param _pop the given population
		 */
		bool operator()(const eoPop < MOEOT > & _pop)
		{
			bool res=false;
			for (unsigned int i=0;i<_pop.size();i++){
				res=operator()(_pop[i])||res;
			}
			return res;
	}



	protected:
		/**
		 * apply a modification
		 * @param _modif the modification to apply
		 **/
		void apply_modif(typename moeoArchiveIndex<MOEOT>::modif &_modif){
			if (_modif.newIdx==-1){
				int oldIdx=size()-1;
				(*this)[_modif.oldIdx]=(*this)[size()-1];
				ObjectiveVector obj=(*this)[_modif.oldIdx].objectiveVector();
				typename moeoArchiveIndex<MOEOT>::modif upd(obj,oldIdx,_modif.oldIdx);
				index.update(upd);
				pop_back();
			}
		}

		//not used yet...
		void apply_modif(std::vector<typename moeoArchiveIndex<MOEOT>::modif> &_modifs){
			unsigned int num_to_delete=0;
			for (unsigned int i=0;i<_modifs.size();i++){
				if (_modifs[i].newIdx==-1){
					num_to_delete++;
					int oldIdx=size()-1;
					(*this)[_modifs[i].oldIdx]=(*this)[size()-1];
					ObjectiveVector obj=(*this)[_modifs[i].oldIdx].objectiveVector();
					typename moeoArchiveIndex<MOEOT>::modif upd(obj,oldIdx,_modifs[i].oldIdx);
					index.update(upd);
				}
			}
			for (unsigned int i=0;i<num_to_delete;i++)
				pop_back();
		}



	private:
		moeoArchiveIndex<MOEOT> &index;



};

#endif /*MOEOINDEXEDARCHIVE_H_*/
