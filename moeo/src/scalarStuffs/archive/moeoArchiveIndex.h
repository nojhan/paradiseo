#ifndef MOEOARCHIVEINDEX_H_
#define MOEOARCHIVEINDEX_H_

#include <eoPop.h>
#include <archive/moeoArchive.h>

/**
 * Inteface for Archive Indexes
 */
template < class MOEOT >
class moeoArchiveIndex
{

	public:
		//type of MOEOT Objective vector
		typedef typename MOEOT::ObjectiveVector ObjectiveVector;

		/**type for a modification that will have to be applied to the archive
		 * each item concern one ObjectiveVector, designated by itemObjective
		 **/
		struct modif{
			public:
				//Objective vector of the concerned item 
			       	ObjectiveVector itemObjective;
				//oldIdx is the index of the item in the vector before the modification (in the archive itself, not in the index)
				int oldIdx;
				//newIdx is the new index of the item in the vector after the modification (in the archive itself, not in the index)
				//-1 if deletion has to occur
				int newIdx;
				/**
				 * ctor for a deletion
				 * @param _obj the objectiveVector of the concerned entry
				 * @param _oldIdx the current index of the concerned entry in the vector (before deletion)
				 */
				modif(ObjectiveVector& _obj, int _oldIdx):itemObjective(_obj),oldIdx(_oldIdx),newIdx(-1){}
				/**
				 * ctor for a move
				 * @param _obj the objectiveVector of the concerned entry
				 * @param _oldIdx the current index of the concerned entry in the vector (before moving)
				 * @param _newIdx the index of the concerned entry in the vector after moving
				 **/
				modif(ObjectiveVector& _obj, int _oldIdx,int _newIdx):itemObjective(_obj),oldIdx(_oldIdx),newIdx(_newIdx){}
		};

		/**
		 * principal method for the index, add a moeot to the index
		 * @param _moeot the MOEOT we try to insert
		 * @param _insert should we really insert the moeot, or just check if we have to
		 * @return a pair, the first is a boolean indicating if the insertion can occur, the second a vector of modification
		 **/
		virtual std::pair<bool,std::vector<modif> > operator()(const MOEOT& _moeot, bool _insert=true)=0;

		/*
		 * method for adding a population of moeot to the the index
		 * @param _pop the population of MOEOT we try to insert
		 * @param _insert should we really insert the moeot, or just check if we have to
		 * @return a pair, the first is how many moeot can be inserted, the second a vector of modification that would have to occur to insert
		 */
//		virtual std::pair<bool,std::vector<modif> > operator()(const eoPop<MOEOT>& _pop, bool _insert=true)=0;

		/**
		 * when updates will be necessary to keep indexes of archive and index synced, the archive will launch this method
		 * @param _update the update to do, see modif documentation
		 * @return false if no problem occured
		 */
		virtual bool update( modif& _update)=0;

		/**
		* creates a modif that move the item ObjectiveVector placed at idx oldIdx in the archive to newIdx, or delete it if newIdx=-1
		* @param _obj the objectiveVector we want to move
		* @param _oldIdx the index of the item we want to move in the vector
		* @param _newIdx the new index for the item, -1 if we want it deleted
		**/
		static modif make_modif(ObjectiveVector &_obj,int _oldIdx,int _newIdx=-1){
			modif res(_obj,_oldIdx,_newIdx);
			return res;
		}



};

#endif /*MOEOARCHIVEINDEX_H_*/
