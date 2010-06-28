#ifndef MOEOQUICKUNBOUNDEDARCHIVEINDEX_H_
#define MOEOQUICKUNBOUNDEDARCHIVEINDEX_H_

#include <eoPop.h>
#include <archive/moeoArchive.h>
#include <comparator/moeoObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <algorithm>
#include <iostream>

/**
 * Archive used for 2 dimension vectors which remove pareto dominated values
 * the index is ordered following the first objective
 */
template < class MOEOT >
class moeoQuickUnboundedArchiveIndex : public moeoArchiveIndex < MOEOT >
{

	public:



		/**
		 * The type of an objective vector for a solution
		 */
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;
		typedef typename moeoArchiveIndex<MOEOT>::modif modif;
//		typedef typename moeoArchiveIndex < MOEOT> :: s_update s_update;

		/**
		 * Default ctor. Pareto !!!!
		 * The moeoObjectiveVectorComparator used to compare solutions is based on Pareto dominance
		 */
		moeoQuickUnboundedArchiveIndex() : index() {}

		/**
		 * Ctor
		 * @param _comparator the moeoObjectiveVectorComparator used to compare solutions
		 */
		//moeoQuickUnboundedArchive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator) : moeoArchive < MOEOT >(_comparator),index() {}

		/**struct for an entry of the index
		 * obj is the objective vector of the vector[indice]
		 */
		struct entree{
			entree(ObjectiveVector _obj, int _indice):obj(_obj),indice(_indice){}
			bool operator == (const entree a){
				return obj==a.obj;
			}
			ObjectiveVector obj;
			int indice;
		};
		/**
		 * equivalent to "number one element should be on top of number two element" in the list by looking to the first obj
		 */
		struct CompareByFirst
			: std::binary_function< bool, entree, entree > {
				bool operator ()(
						const entree& elem1,
						const entree& elem2
						) const {
					if (ObjectiveVector::minimizing(0)){
						return elem1.obj[0] > elem2.obj[0];
					}
					else{
						return elem1.obj[0] < elem2.obj[0];
					}
				}
			}cbf; 
		/**
		 * equivalent to "number one element should be on top of number two element" in the list by looking to the 2nd obj
		 */
		struct CompareByLast
			: std::binary_function< bool, entree, entree > {
				bool operator ()(
						const entree& elem1,
						const entree& elem2
						) const {
					if (ObjectiveVector::minimizing(1)){
						return elem1.obj[1] < elem2.obj[1];
					}
					else{
						return elem1.obj[1] > elem2.obj[1];
					}
				}
			}cbl; 


		struct CompareByLast2
			: std::binary_function< bool, MOEOT, MOEOT > {
				bool operator ()(
						const MOEOT& elem1,
						const MOEOT& elem2
						) const {
					if (ObjectiveVector::minimizing(1)){
						return elem1.objectiveVector()[1] < elem2.objectiveVector()[1];
					}
					else{
						return elem1.objectiveVector()[1] > elem2.objectiveVector()[1];
					}
				}
			}cbl2; 
		/**
		 * type for the index
		 */
		typedef typename std::set<entree,CompareByLast> MOEOTIndex;
		/**
		 * iterator from the index
		 */
		typedef typename std::set<entree,CompareByLast>::iterator MOEOTIndexIte;
		/**
		 * iterator for gcc stop being annoying
		 */
		typedef typename std::set<MOEOT>::iterator set_ite;



		/**
		  updates the index following a modif
		  @param _update the modification to apply
		  @return false
		  */
		bool update(modif& _update){
			entree oldEnt(_update.itemObjective,_update.oldIdx);
			entree newEnt(_update.itemObjective,_update.newIdx);
			index.erase(oldEnt);
			index.insert(newEnt);
			return false;
		}

/*
		std::pair<bool,std::vector<modif> > operator()(const eoPop<MOEOT>& _pop, bool _insert=true){
			std::cout<<"OH, HI, je fais quelque chose"<<std::endl;
			std::pair < bool, std::vector<modif> > res;
			res.first=false;
			std::vector <modif> tmp;
			for (unsigned int i=0;i<_pop.size();i++){
				std::cout<<"once va être créé"<<std::endl;
				std::pair<bool,std::vector<modif> > once=operator()(_pop[i],_insert);
				if (once.first){
					std::cout<<"once vrai taille "<<once.second.size()<<std::endl;
					std::copy(once.second.begin(),once.second.end(),res.second.end());
					res.first=true;
				}

			}
			return res;
		};
*/

		virtual std::pair<bool,std::vector<modif> > operator()(const MOEOT& _moeo, bool _insert=true){
			return insert(_moeo,_insert);
		}
		/**
		 * inserts a _moeo in the index
		 * @param _moeo the MOEOT to insert
		 * @param _insert if _insert is false we only ask the index, and dont modify it
		 * @return a pair composed by a boolean indicating if the moeot can be inserted, and a list of modif to do so
		 */
		virtual std::pair<bool,std::vector<modif> > insert(const MOEOT& _moeo, bool _insert=true){
//			std::cout<<"entree dans l'algo avec "<<_moeo.objectiveVector()<<std::endl;
			MOEOTIndexIte it,it2,it4;
			std::pair<bool,std::vector<modif> > res;
			std::vector<entree> to_er;
			res.first=false;
			if (index.empty()){
				std::cout<<"empty donc ok"<<std::endl;
				if (_insert)
					index.insert(entree(_moeo.objectiveVector(),index.size()));
				res.first=true;
				return res;  
			}
			it=index.lower_bound(entree(_moeo.objectiveVector(),-1));
			if (it==index.end()) {
				it--;
				if (!comparator(_moeo.objectiveVector(),(*it).obj)){
					std::cout<<"fin et ok"<<std::endl;
					if (_insert)
						index.insert(entree(_moeo.objectiveVector(),index.size()));
					res.first=true;
				}else {
					std::cout<<"fin et ko"<<std::endl;
				}
				return res;
			}
			if ((_moeo.objectiveVector()==(*it).obj) or  (comparator(_moeo.objectiveVector(),(*it).obj))){
				std::cout<<"middle ko bas"<<std::endl;
				return res;
			}
			if (it!=index.begin()){
				it2=it;
				it2--;
				if (comparator(_moeo.objectiveVector(),(*it2).obj)){
					std::cout<<"middle ko haut"<<std::endl;
					return res;
				}
			}

			it2=it;
			while (it2!=index.end() && comparator((*it2).obj,_moeo.objectiveVector())){ 
				it2++;
			}
			for (it4=it;it4!=it2;it4++){
				std::cout<<"ajout d'un truc à del"<<std::endl;
				ObjectiveVector cpy=(*it4).obj;
				int cpy_idx=(*it4).indice;
				modif new_modif(cpy,cpy_idx);
				res.second.push_back(new_modif);
				to_er.push_back(*it4);
			}
			if (_insert){
				for (unsigned int i=0;i<to_er.size();i++){
					index.erase(to_er[i]);
				}
				index.insert(entree(_moeo.objectiveVector(),index.size()));
			}
			res.first=true;
			std::cout<<"sortie avec insertion"<<std::endl;
			return res;
		}


	protected:



	private:
		MOEOTIndex index;
		moeoParetoObjectiveVectorComparator<ObjectiveVector> comparator;



};

#endif /*MOEOQUICKUNBOUNDEDARCHIVE_H_*/
