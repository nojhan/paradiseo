/***************************************************************************
 *   Copyright (C) 2008 by Waldo Cancino   *
 *   wcancino@icmc.usp.br   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef MOEOOBJVECSTAT_H_
#define MOEOOBJVECSTAT_H_

#include <utils/eoStat.h>

#if  defined(_MSC_VER) && (_MSC_VER < 1300)
template <class MOEOT>
class moeoObjVecStat : public eoStat<MOEOT, MOEOT::ObjectiveVector>
#else
template <class MOEOT>
class moeoObjVecStat : public eoStat<MOEOT, typename MOEOT::ObjectiveVector>
#endif
{
	public:
    using eoStat<MOEOT, typename MOEOT::ObjectiveVector>::value;

    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
	typedef typename MOEOT::ObjectiveVector::Traits Traits;

    moeoObjVecStat(std::string _description = "")
        : eoStat<MOEOT, ObjectiveVector>(ObjectiveVector(), _description)
        {}

    virtual void operator()(const eoPop<MOEOT>& _pop) { doit(_pop); };
	private:
	virtual void doit(const eoPop<MOEOT> &_pop) = 0;
};

template <class MOEOT>
class moeoBestObjVecStat : public moeoObjVecStat<MOEOT>
{
public:

	using moeoObjVecStat<MOEOT>::value;
    typedef typename moeoObjVecStat<MOEOT>::ObjectiveVector ObjectiveVector;


    moeoBestObjVecStat(std::string _description = "Best ")
        : moeoObjVecStat<MOEOT>(_description)
        {}


    virtual std::string className(void) const { return "moeoBestObjVecStat"; }

	const MOEOT & bestindividuals(unsigned int objective) { return *(best_individuals[objective]); }

private :

	std::vector<typename eoPop<MOEOT>::const_iterator> best_individuals;

    struct CmpObjVec
    {
      CmpObjVec(unsigned _which, bool _maxim) : which(_which), maxim(_maxim) {}

      bool operator()(const MOEOT& a, const MOEOT& b)
      {
        if (maxim)
          return a.objectiveVector()[which] < b.objectiveVector()[which];

        return a.objectiveVector()[which] > b.objectiveVector()[which];
      }

      unsigned which;
      bool maxim;
    };

    // Specialization for objective vector
    void doit(const eoPop<MOEOT>& _pop)
    {
      typedef typename moeoObjVecStat<MOEOT>::Traits traits;
	  
      value().resize(traits::nObjectives());
	  
      for (unsigned o = 0; o < traits::nObjectives(); ++o)
      {
        typename eoPop<MOEOT>::const_iterator it = std::max_element(_pop.begin(), _pop.end(), CmpObjVec(o, traits::maximizing(o)));
		value()[o] = it->objectiveVector()[o];
		best_individuals.push_back( it );
      }
    }
    // default
};

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
