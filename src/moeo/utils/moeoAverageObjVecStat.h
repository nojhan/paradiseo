/*
* <moeoComparator.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2009
* 
*
* Waldo Cancino, Arnaud Liefooghe
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

#ifndef MOEOAVERAGEOBJVECSTAT_H_
#define MOEOAVERAGEOBJVECSTAT_H_

#include "moeoObjVecStat.h"

/** Calculates  average scores for each objective 
*/


template <class MOEOT> class moeoAverageObjVecStat : public moeoObjVecStat<MOEOT>
{
public :

    using moeoObjVecStat<MOEOT>::value;
    typedef typename moeoObjVecStat<MOEOT>::ObjectiveVector ObjectiveVector;


    moeoAverageObjVecStat(std::string _description = "Average Objective Vector")
      : moeoObjVecStat<MOEOT>(_description) {}


    virtual std::string className(void) const { return "moeoAverageObjVecStat"; }

private :

	typedef typename MOEOT::ObjectiveVector::Type Type;
    
	template<typename T> struct sumObjVec
    {
      sumObjVec(unsigned _which) : which(_which) {}

      Type operator()(Type &result, const MOEOT& _obj)
      {
			return result + _obj.objectiveVector()[which];
      }

      unsigned which;
    };

    // Specialization for pareto fitness
    void doit(const eoPop<MOEOT>& _pop)
    {
      typedef typename moeoObjVecStat<MOEOT>::Traits traits;
      value().resize(traits::nObjectives());

      for (unsigned o = 0; o < value().size(); ++o)
      {
		value()[o] = std::accumulate(_pop.begin(), _pop.end(), Type(), sumObjVec<Type>(o));
        value()[o] /= _pop.size();
      }
    }
};

#endif
