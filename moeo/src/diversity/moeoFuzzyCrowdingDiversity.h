/*
  <moeoFuzzyCrowdingDiversity.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------
#ifndef MOEOFUZZYCROWDINGDIVERSITY_H_
#define MOEOFUZZYCROWDINGDIVERSITY_H_

#include <eoPop.h>
#include <diversity/moeoDiversityAssignment.h>
#include <comparator/moeoOneObjectiveComparator.h>

#include <comparator/moeoFuzzyParetoComparator.h>
#include <distance/moeoExpectedFuzzyDistance.h>



/**
 * Diversity assignment sheme based on crowding proposed in:
 * K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2 (2002).
 * Tis strategy assigns diversity values FRONT BY FRONT. It is, for instance, used in NSGA-II.
 */
template < class MOEOT >
class moeoFuzzyCrowdingDiversity : public CrowdingDiversityAssignment < MOEOT >
  {
  public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
      std::cout << "WARNING : updateByDeleting not implemented in FrontByFrontCrowdingDistanceDiversityAssignment" << std::endl;
    }

  private:

    using CrowdingDiversityAssignment < MOEOT >::inf;
    using CrowdingDiversityAssignment < MOEOT >::tiny;

    /**
     * Sets the distance values
     * @param _pop the population
     */

    void setDistances (eoPop <MOEOT> & _pop)
    {
      unsigned int a,b;
      double min, max, distance;
      unsigned int nObjectives = MOEOT::ObjectiveVector::nObjectives();
      // set diversity to 0 for every individual
      for (unsigned int i=0; i<_pop.size(); i++)
        {
          _pop[i].diversity(0.0);
        }
      // sort the whole pop according to fitness values
      moeoFitnessThenDiversityComparator < MOEOT > fitnessComparator;
	  std::vector<MOEOT *> sortedptrpop;
	  sortedptrpop.resize(_pop.size());
	  // due to intensive sort operations for this diversity assignment,
	  // it is more efficient to perform sorts using only pointers to the
      // population members in order to avoid copy of individuals
	  for(unsigned int i=0; i< _pop.size(); i++) sortedptrpop[i] = & (_pop[i]);
      //sort the pointers to population members
	  moeoFuzzyParetoComparator<MOEOT> comp( fitnessComparator);
	  std::sort(sortedptrpop.begin(), sortedptrpop.end(), comp);
      // compute the expected crowding distance values for every individual "front" by "front" (front : from a to b)
      a = 0;	        			// the front starts at a
      while (a < _pop.size())
        {
		  b = lastIndex(sortedptrpop,a);	// the front ends at b
          //b = lastIndex(_pop,a);	// the front ends at b
          // if there is less than 2 individuals in the front...
          if ((b-a) < 2)
            {
              for (unsigned int i=a; i<=b; i++)
                {
					sortedptrpop[i]->diversity(inf());
                  //_pop[i].diversity(inf());
                }
            }
          // else...
          else
            {
              // for each objective
              for (unsigned int obj=0; obj<nObjectives; obj++)
                {
                  // sort in the descending order using the values of the objective 'obj'
                  moeoOneObjectiveComparator < MOEOT > objComp(obj);
				  moeoFuzzyParetoComparator<MOEOT> comp( objComp );
				  std::sort(sortedptrpop.begin(), sortedptrpop.end(), comp);
                  // min & max
                  min = (sortedptrpop[b])->objectiveVector()[obj].second;
                  max = (sortedptrpop[a])->objectiveVector()[obj].second;

                  // avoid extreme case
                  if (min == max)
                    {
                      min -= tiny();
                      max += tiny();
                    }
                  // set the diversity value to infiny for min and max
                  sortedptrpop[a]->diversity(inf());
                  sortedptrpop[b]->diversity(inf());
                  // set the diversity values for the other individuals
                  for (unsigned int i=a+1; i<b; i++)
                    {   
                        distance = ( sortedptrpop[i-1]->moeoExpectedFuzzyDistance(objectiveVector()[obj]) - sortedptrpop[i+1]->moeoExpectedFuzzyDistance(objectiveVector()[obj] )) / (max-min);
                        sortedptrpop[i]->diversity(sortedptrpop[i]->diversity() + distance);
                    }
                }
            }
          // go to the next front
          a = b+1;
        }
    }



    /**
     * Returns the index of the last individual having the same fitness value than _pop[_start]
     * @param _pop the vector of pointers to population individuals
     * @param _start the index to start from
     */

    unsigned int lastIndex (std::vector<MOEOT *> & _pop, unsigned int _start)
    {
      unsigned int i=_start;
      while ( (i<_pop.size()-1) && (_pop[i]->fitness()==_pop[i+1]->fitness()) )
        {
          i++;
        }
      return i;
    }


  };

#endif /*MOEOFUZZYCROWDINGDIVERSITY_H_*/
