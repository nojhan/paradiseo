// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// moeoComparator.h
// (c) OPAC Team (LIFL), Dolphin Project (INRIA), 2007
/*
    This library...

    Contact: paradiseo-help@lists.gforge.inria.fr, http://paradiseo.gforge.inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef MOEOCOMPARATOR_H_
#define MOEOCOMPARATOR_H_

#include <eoFunctor.h>
#include <eoPop.h>

/**
 * Functor allowing to compare two solutions
 */
template < class MOEOT > class moeoComparator:public eoBF < const MOEOT &, const MOEOT &,
  const bool >
{
public:
 // virtual const bool operator () (const MOEOT & _moeo1, const MOEOT & _moeo){}
};


/**
 * Functor allowing to compare two solutions according to their first objective value, then their second, and so on
 */
template < class MOEOT > class moeoObjectiveComparator:public moeoComparator <
  MOEOT >
{
public:
	/**
	 * Returns true if _moeo1 is smaller than _moeo2 on the first objective, then on the second, and so on
	 * @param _moeo1 the first solution
	 * @param _moeo2 the second solution
	 */
  const bool operator () (const MOEOT & _moeo1, const MOEOT & _moeo2)
  {
    return _moeo1.objectiveVector () < _moeo2.objectiveVector ();
  }
};


/**
 * Functor allowing to compare two solutions according to their fitness values
 */
//template < class MOEOT >
//class moeoFitnessComparator : public moeoComparator < MOEOT >
//{
//public:
//      /**
//       * Returns true if the fitness value of _moeo1 is smaller than the fitness value of _moeo2
//       * @param _moeo1 the first solution
//       * @param _moeo2 the second solution
//       */
//      const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
//      {
//              return _moeo1.fitness() < _moeo2.fitness();
//      }
//};
//
//
///**
// * Functor allowing to compare two solutions according to their diversity values
// */
//template < class MOEOT >
//class moeoDiversityComparator : public moeoComparator < MOEOT >
//{
//public:
//      /**
//       * Returns true if the diversity value of _moeo1 is smaller than the diversity value of _moeo2
//       * @param _moeo1 the first solution
//       * @param _moeo2 the second solution
//       */
//      const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
//      {
//              return _moeo1.diversity() < _moeo2.diversity();
//      }
//};


/**
 * Functor allowing to compare two solutions according to their fitness values, then according to their diversity values
 */
template < class MOEOT > class moeoFitnessThenDiversityComparator:public moeoComparator <
  MOEOT >
{
public:
	/**
	 * Returns true if _moeo1 is smaller than _moeo2 according to their fitness values, then according to their diversity values	 
	 * @param _moeo1 the first solution
	 * @param _moeo2 the second solution
	 */
  const bool operator () (const MOEOT & _moeo1, const MOEOT & _moeo2)
  {
    if (_moeo1.fitness () == _moeo2.fitness ())
      {
	return _moeo1.diversity () < _moeo2.diversity ();
      }
    else
      {
	return _moeo1.fitness () < _moeo2.fitness ();
      }
  }
};


/**
 * Functor allowing to compare two solutions according to their diversity values, then according to their fitness values
 */
template < class MOEOT > class moeoDiversityThenFitnessComparator:public moeoComparator <
  MOEOT >
{
public:
	/**
	 * Returns true if _moeo1 is smaller than _moeo2 according to their diversity values, then according to their fitness values
	 * @param _moeo1 the first solution
	 * @param _moeo2 the second solution
	 */
  const bool operator () (const MOEOT & _moeo1, const MOEOT & _moeo2)
  {
    if (_moeo1.diversity () == _moeo2.diversity ())
      {
	return _moeo1.fitness () < _moeo2.fitness ();
      }
    else
      {
	return _moeo1.diversity () < _moeo2.diversity ();
      }
  }
};


/**
 * Functor allowing to compare two solutions according to Pareto dominance relation => USEFULL ???
 *
template < class MOEOT >
class moeoParetoDominanceComparator : public moeoComparator < MOEOT >
{
public:
	/**
	 * Returns true if _moeo1 is dominated by _moeo2 according to Pareto dominance relation
	 * @param _moeo1 the first solution
	 * @param _moeo2 the second solution
	 *
	const bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
	{
		bool result = false;
		typedef typename MOEOT::ObjectiveVector::Traits ObjectiveVectorTraits;
		for (unsigned i=0; i<ObjectiveVectorTraits::nObjectives(); i++)
		{
			// first, we have to check if the 2 objective values are not equal on the ith objective
			if ( fabs(_moeo1.objectiveVector()[i] - _moeo2.objectiveVector()[i]) > ObjectiveVectorTraits::tolerance() )
			{
				// if the ith objective have to be minimized...
				if (ObjectiveVectorTraits::minimizing(i))
				{
					if (_moeo1.objectiveVector()[i] < _moeo2.objectiveVector()[i])
					{
						return false;	// _moeo2 cannot dominate _moeo1
					}
					result = true;
				}
				// if the ith objective have to be maximized...
				else if (ObjectiveVectorTraits::maximizing(i))
				{
					if (_moeo1.objectiveVector()[i] > _moeo2.objectiveVector()[i])
					{
						return false;	// _moeo2 cannot dominate _moeo1
					}
					result = true;
				}
			}
		}
		return result;
	}
};
*/

#endif /*MOEOCOMPARATOR_H_ */
