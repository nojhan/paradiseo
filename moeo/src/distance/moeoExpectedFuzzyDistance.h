/*
  <moeoExpectedFuzzyDistance.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOEXPECTEDFUZZYDISTANCE_H_
#define MOEOEXPECTEDFUZZYDISTANCE_H_

#include <math.h>
#include <distance/moeoObjSpaceDistance.h>
#include <utils/moeoFuzzyObjectiveVectorNormalizer.h>

/**
 * An expected euclidian distance between two fuzzy solutions in the objective space 
 * Every solution value is expressed by a triangular fuzzy number 
 */
template < class MOEOT>
class moeoExpectedFuzzyDistance : public moeoObjSpaceDistance < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    /** the fitness type of the solutions */
    typedef typename MOEOT::Fitness Fitness;

    /**
      default ctr
      */
    moeoExpectedFuzzyDistance ()
	  {}

    /**
     * tmp1 and tmp2 take the Expected values of Objective vectors
     * Returns the expected distance between _obj1 and _obj2 in the objective space
     * @param _obj1 the first objective vector
     * @param _obj2 the second objective vector
     */
const Fitness operator()(const ObjectiveVector & _obj1, const ObjectiveVector & _obj2)
    {
      Fitness result = 0.0;
      Fitness tmp1, tmp2;
      for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
        {

          tmp1 =  ((_obj1)[i].first + (_obj1)[i].third + 2 *(_obj1)[i].second ) /4 ; 
          tmp2 = ((_obj2)[i].first + (_obj2)[i].third + 2* (_obj2)[i].second ) /4 ;


          result += (tmp1-tmp2) * (tmp1-tmp2);
        }
      return sqrt(result);
    }


  };

#endif /*MOEOEXPECTEDFUZZYDISTANCE_H_*/
