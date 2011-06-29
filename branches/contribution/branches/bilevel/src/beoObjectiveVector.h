/*
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2010
*
* Legillon Francois
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/
//-----------------------------------------------------------------------------
//Objective Vector for bilevel problems
#ifndef BEO_OBJECTIVEVECTOR_H_
#define BEO_OBJECTIVEVECTOR_H_
#include <vector>
#include <core/moeoObjectiveVectorTraits.h>
template <class ObjectUp, class ObjectLow> class beoObjectiveVector{

	public:
		typedef ObjectUp ObjUp;
		typedef ObjectLow ObjLow;
		typedef moeoObjectiveVectorTraits Traits;
		beoObjectiveVector():regle(false){}



		/**
		 * doesnt do a thing
		 */
		static void setup(unsigned int nobj,std::vector<bool> &_bobjectives){
		}

		/**
		 * return the size of the mode flag level objective
		 */
		unsigned int size()const {
			if(mode())
				return up().size();
			else
				return low().size();
		}
		/**
		 * return the [] result of the mode flag level variable
		 **/
		double & operator[](unsigned int i){
			if (mode())
				return up()[i];
			else{
				return low()[i];
			}
		}
		/**
		 * return the [] result of the mode flag level variable
		 **/
		const double & operator[](unsigned int i)const {
			if (mode())
				return up()[i];
			else{
				return low()[i];
			}
		}
	
		/**
		 * return the number of objective of the lower level
		 */
		static unsigned int nObjectives(){
				return ObjLow::nObjectives();
		}
		/**
		 * return true if the lower level is to be minimized
		 */
		static bool minimizing(unsigned int i){
				return ObjLow::minimizing(i);
		}
		/**
		 * return true if the lower level is to be maximized
		 */
		static bool maximizing(unsigned int i){
				return ObjLow::maximizing(i);
		}

		/**
		 * return the upper part of the objective
		 **/
		ObjectUp &up(){
			return mup;
		}
		/**
		 * return the lower part of the objective
		 **/
		ObjectLow &low(){
			return mlow;
		}
		/**
		 * return the upper part of the objective
		 **/
		const ObjectUp &up()const{
			return mup;
		}
		/**
		 * return the lower part of the objective
		 **/
		const ObjectLow &low()const{
			return mlow;
		}
		/**
		 * return the upper part of the objective
		 **/
		void upset(const ObjectUp &_up){
			mup=_up;
		}
		/**
		 * sets the lower part of the objective
		 **/
		void lowset(const ObjectLow &_low){
			regle=true;
			mlow=_low;
		}
		/**
		 * returns the mode flag
		 */
		bool mode() const{
			return false;
		}
		/**
		 * sets the mode flag
		 */
		void mode(bool _mode){
			flag=_mode;
		}
		/**
		 * return true if different
		 */
		bool operator!=(const beoObjectiveVector &_vec)const{
			return !operator==(_vec);
		}
		/**
		 * return true if both of the level are equal
		 */
		bool operator==(const beoObjectiveVector &_vec)const{
			return _vec.up()==up() && _vec.low()==low();
		}



	private:
		ObjectUp mup;
		ObjectLow mlow;
		bool flag;
		bool regle;
};
template <class a, class b> std::istream & operator>>(std::istream & _is,beoObjectiveVector<a,b> &_be){
	return _is;
}
template <class a, class b> std::ostream & operator<<(std::ostream & _os,const beoObjectiveVector<a,b> &_be){
	return _os;
}
#endif
