/*
  <moeoBertDistance.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOBERTDISTANCE_H_
#define MOEOBERTDISTANCE_H_

#include <math.h>
#include <distance/moeoObjSpaceDistance.h>
#include "ObjectiveVectorNormalizer.h"

/**
 * A class allowing to compute an Bert distance between two fuzzy solutions in the objective space 
 with normalized objective values (i.e. between 0 and 1).
 * A distance value then lies between 0 and sqrt(nObjectives).
 */
template < class MOEOT>
class moeoBertDistance : public moeoObjSpaceDistance < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;
    /** the fitness type of the solutions */
    typedef typename MOEOT::Fitness Fitness;

    /**
      default ctr
      */
   /* moeoBertDistance ()
	  {}*/
	/**
      ctr with a normalizer
      @param _normalizer the normalizer used for every ObjectiveVector
      */
    /**
      default ctr
      */
    moeoBertDistance ():normalizer(defaultNormalizer)
	  {}


	double calculateBertDistance(std::triple<double, double, double> A, std::triple<double, double, double> B)
	{
		double midA = 0.5 * (A.first + A.third);
		double midB = 0.5 * (B.first + B.third);
		double sprA = 0.5 * (A.first - A.third);
		double sprB = 0.5 * (B.first - B.third);

		double theta = 0.5;

		return sqrt((midA -midB) * (midA -midB) + theta * (sprA - sprB) * (sprA - sprB));
	}


    /**
     * tmp1 and tmp2 take the Expected values of Objective vectors
     * Returns the Bert distance between _obj1 and _obj2 in the objective space
     * @param _obj1 the first objective vector
     * @param _obj2 the second objective vector
     */
const Fitness operator()(const ObjectiveVector & _obj1, const ObjectiveVector & _obj2)
    {
		vector<double> v_BD;
		double dist=0.0;

		for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
		{
			dist +=calculateBertDistance(normalizer(_obj1)[i], normalizer(_obj2)[i]);
		}
		
		//dist += normalizer(v_BD);

		return dist/ObjectiveVector::nObjectives();

}


  private:
	  ObjectiveVectorNormalizer<MOEOT> Normalizer;



  };

#endif /*MOEOBERTDISTANCE_H_*/
