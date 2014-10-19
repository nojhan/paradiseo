/*
* <moeoIncrEvalSingleObjectivizer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2009
* (C) OPAC Team, LIFL, 2002-2007
*
* François Legillon 
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

#ifndef MOEOINCREVALSINGLEOBJECTIVIZER_H_
#define MOEOINCREVALSINGLEOBJECTIVIZER_H_

//#include <moeo>
//#include <moMoveIncrEval.h> // ?
#include "../../fitness/moeoFitnessAssignment.h"
#include "../../fitness/moeoSingleObjectivization.h"
#include "../../../eo/eoEvalFunc.h"

/**
 * Class to adapt mo algorithms (moTS moVNC...) for multiobjectives
 * This class play a moMoveIncrEval but can be used with multi objectives
 * Use a Singleobjectivizer to set the fitness value according to each dimension
 */
template < class MOEOT , class Move  >
class moeoIncrEvalSingleObjectivizer : public moeoSingleObjectivization<MOEOT>, public moMoveIncrEval < Move, typename MOEOT::Fitness>
  {
  public:

    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    moeoIncrEvalSingleObjectivizer (){}
    /**
      Constructor
      @param _singler a singleObjectivizer to calculte the fitness from the objectiveVector
      @param _incr incremental evaluation of moeots
     */
	  moeoIncrEvalSingleObjectivizer ( moeoSingleObjectivization<MOEOT> &_singler, moMoveIncrEval<Move,typename MOEOT::ObjectiveVector> &_incr):
		  singler(_singler), incr(_incr)
	  {}
    /**
     * herited from moeoFitnessAssignment, calculate fitness for all population
     * @param _pop the population
     */
    virtual void operator () (eoPop < MOEOT > & _pop){
	    singler(_pop);
    };

    /**
      herited from eoEvalFunc, calculate fitness for a moeot      
      @param _moeot
     */
    virtual void operator() (MOEOT & _moeot){
	    singler(_moeot);
    };

    /**
      calculate fitness from an objectiveVector
      @param _moeot a valid ObejctiveVector
      @return the fitness value for the objectiveVector
     */
    virtual typename MOEOT::Fitness operator() (const typename MOEOT::ObjectiveVector & _moeot){
	    return singler(_moeot);
    };
    /**
      evaluates incrementally the fitness for a moeo 
     @param _mov a movement to virtually apply to _moeo
     @param _moeo the base solution
     @return the fitness of _moeo with _move applied
     */
    virtual typename MOEOT::Fitness operator() ( const Move &_mov, const MOEOT &_moeo ){
	    return singler(incr_obj(_mov,_moeo));
    }

    /**
      evaluates incrementally the objectiveVector for a moeo
     @param _mov a movement to virtually apply to _moeo
     @param _moeo the base solution
     @return the objectiveVector of _moeo with _move applied
      */
    virtual ObjectiveVector incr_obj ( const Move &_mov, const MOEOT &_moeo ){
	    return incr(_mov,_moeo);
    }
    /** dummy method**/
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec){}
  private: 
    moeoSingleObjectivization<MOEOT> &singler; 
    moMoveIncrEval<Move,typename MOEOT::ObjectiveVector> &incr;
  };

#endif /*MOEOINCREVALSINGLEOBJECTIVIZER_H_*/
