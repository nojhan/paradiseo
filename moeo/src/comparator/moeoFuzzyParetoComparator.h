/*
  <moeoFuzzyParetoComparator.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------


#ifndef MOEOFUZZYPARETOCOMPARATOR_H_
#define MOEOFUZZYPARETOCOMPARATOR_H_

#include <comparator/moeoObjectiveVectorComparator.h>
//#include "triple.h"

/**
 * This  class allows ranking fuzzy-valued objectives according to new dominance relations.  
   The dominance is defined between vectors of triangular fuzzy numbers (each number is expressed by a triplet of values [first, second, third].
 */
template < class ObjectiveVector >
class moeoFuzzyParetoObjectiveVectorComparator : public moeoObjectiveVectorComparator < ObjectiveVector >
  {
	  int compareTriangNum (std::triple<double, double, double> A, std::triple<double, double, double> B)
{
	// Total dominance
	if (A.third < B.first) return TNCR_TOTAL_DOMINANCE;		


	// Partial Strong dominance
	if (A.third >= B.first && A.second <= B.first && A.third <= B.second) return TNCR_PARTIAL_STRONG_DOMINANCE;		



	// Partial Weak dominance
	if ((A.first <B.first && A.third < B.third) &&		
		( (A.second <= B.first && A.third > B.second) || (A.second > B.first && A.third <= B.second ) || (A.second > B.first && A.third > B.second )))
		 
		 return TNCR_PARTIAL_WEAK_DOMINANCE;

	if (A.first < B.first && A.third >= B.third && A.second < B.second) 
		return TNCR_PARTIAL_WEAK_DOMINANCE;		
	else if (A.first < B.first && A.third >= B.third && A.second >= B.second && (B.first - A.first) > (B.third - A.third)) 
		return TNCR_PARTIAL_WEAK_DOMINANCE;		
	
	return 0;
}
  public:

    /**
     * Returns true if _objectiveVector1 V1 is dominated by _objectiveVector2 V2 means  V2 dominates V1
     * @param _objectiveVector1 the first objective vector
     * @param _objectiveVector2 the second objective vector	
     */
    /*const*/ bool operator()(const ObjectiveVector & _objectiveVector1, const ObjectiveVector & _objectiveVector2)
{
	bool dom = false;
	int nb_Different_Objective_Values = 0,
	nb_Total_Dominance = 0,
	nb_Partial_Strong_Dominance = 0,
	nb_Partial_Weak_Dominance = 0,
	nb_Other = 0;
	
	//nObjective= 2
	for (unsigned int i=0; i<ObjectiveVector::nObjectives(); i++)
	{
		// first, we have to check if the 2 objective values are not equal for the ith objective
		if (( fabs(_objectiveVector1[i].first - _objectiveVector2[i].first) > ObjectiveVector::Traits::tolerance())||
			( fabs(_objectiveVector1[i].second - _objectiveVector2[i].second) > ObjectiveVector::Traits::tolerance())||
			( fabs(_objectiveVector1[i].third - _objectiveVector2[i].third) > ObjectiveVector::Traits::tolerance()))
		{
			nb_Different_Objective_Values++;
			
			// if the ith objective have to be minimized...
			if (ObjectiveVector::minimizing(i))
			{
				if		( compareTriangNum(_objectiveVector2[i], _objectiveVector1[i] ) == TNCR_TOTAL_DOMINANCE ) 			nb_Total_Dominance++;
				else if ( compareTriangNum(_objectiveVector2[i], _objectiveVector1[i] ) == TNCR_PARTIAL_STRONG_DOMINANCE )	nb_Partial_Strong_Dominance++;
				else if ( compareTriangNum(_objectiveVector2[i], _objectiveVector1[i] ) == TNCR_PARTIAL_WEAK_DOMINANCE )	nb_Partial_Weak_Dominance++;
				else 																										nb_Other++;
			}
			else
			{
				// Develop the maximizing compareTriangNum version 
			}
		}
	}
	
	// Strong Pareto Dominance
	if ( nb_Different_Objective_Values == nb_Total_Dominance ||
		 nb_Different_Objective_Values == nb_Partial_Strong_Dominance ||
		 nb_Total_Dominance >= 1 && nb_Partial_Strong_Dominance > 0 ||
		 nb_Total_Dominance >= 1 || nb_Partial_Strong_Dominance >= 1 && nb_Partial_Weak_Dominance > 0)
		 { dom = true; 
		 }
		 
		 
	else if ( nb_Different_Objective_Values == nb_Partial_Weak_Dominance ) { dom = true; }
	else {return false; 
	}

	return dom;
}

enum TriangularNumberComparaisonResult
{
	TNCR_TOTAL_DOMINANCE,
	TNCR_PARTIAL_STRONG_DOMINANCE,
	TNCR_PARTIAL_WEAK_DOMINANCE
};


  };
#endif /*MOEOFUZZYPARETOCOMPARATOR_H_*/
