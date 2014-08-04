/*
* <moeoSingleObjectivization.h>
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

#ifndef MOEOSINGLEOBJECTIVIZATION_H_
#define MOEOSINGLEOBJECTIVIZATION_H_

#include "moeoFitnessAssignment.h"
#include "../../eo/eoEvalFunc.h"
/**
 * Class to adapt multiobjective problems to single objective algorithms
 */
template < class MOEOT >
class moeoSingleObjectivization : public moeoFitnessAssignment < MOEOT > , public eoEvalFunc < MOEOT >
  {
  public:

    /**
     * herited from moeoFitnessAssignment 
     * @param _pop the population
     */
    virtual void operator () (eoPop < MOEOT > & _pop)=0;

    /**
      herited from eoEvalFunc      
      @param _moeot
     */
    virtual void operator() (MOEOT & _moeot)=0;

    virtual typename MOEOT::Fitness operator() (const typename MOEOT::ObjectiveVector & _obj)=0;
	    
  };

#endif /*MOEOSINGLEOBJECTIVIZATION_H_*/
