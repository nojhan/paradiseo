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
//BEO is a class to represent a bilevel problem solution from 
//the solution representation of each level
#ifndef BEO_H_
#define BEO_H_
#include <EO.h>
#include <core/MOEO.h>
#include <beoObjectiveVector.h>
template <
class BEOU, 
      class BEOL, 
      class OBJECTIVE=beoObjectiveVector<typename BEOU::ObjectiveVector, typename BEOL::ObjectiveVector>,
      class Fitness=double
      > 
      class BEO: public MOEO<OBJECTIVE>{
	      public:
		      typedef BEOU U;
		      typedef BEOL L;

		      BEO(BEOU &_up, BEOL &_low, bool _mode=true):up(_up),low(_low),mode(_mode){}
		      BEO(bool _mode=true):mode(_mode){}


		      /**
		       * return the mode flag
		       */
		      bool getMode()const {
			      return mode;
		      }
		      /**
		       * sets the mode flag and change the general objective vector to the corresponding
		       * level
		       */
		      void setMode(bool _up){
			      mode=_up;
			      if (!MOEO<OBJECTIVE>::invalidObjectiveVector()){
				      OBJECTIVE tmp=MOEO<OBJECTIVE>::objectiveVector();
				      tmp.mode(_up);
				      objectiveVector(tmp);
			      }
		      }


		      /**
		       * returns the upper part of the solution
		       */
		      U &upper(){
			      return up;
		      }
		      /**
		       * returns the upper part of the solution
		       */
		      const U &upper() const{
			      return up;
		      }
		      /**
		       * returns the lower part of the solution
		       */
		      L &lower(){
			      return low;
		      }
		      /**
		       * returns the lower part of the solution
		       */
		      const L &lower() const{
			      return low;
		      }
		      /**
		       * returns true if both level are equal
		       */
		      bool operator==(const BEO& _beo)const{
			      return _beo.lower()==lower() && _beo.upper()==upper();
		      }
		      /**
		       * comparator based on the general fitness
		       */
		      bool operator<(const BEO& _beo)const{
			      return MOEO<OBJECTIVE>::fitness()<_beo.fitness();
		      }

		      /**
		       * comparator based on the general fitness
		       */
		      bool operator>(const BEO& _beo)const{
			      return MOEO<OBJECTIVE>::fitness()>_beo.fitness();
		      }

		      void printOn(std::ostream &_os) const{
			      _os<<upper().fitness()<<'\t'<<lower().fitness();
		      }


	      private:
		      U up;
		      L low;
		      bool mode;

      };
#endif
