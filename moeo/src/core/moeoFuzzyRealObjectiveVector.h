/*
  <moeoFuzzyRealObjectiveVector.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOFUZZYREALOBJECTIVEVECTOR_H_
#define MOEOFUZZYREALOBJECTIVEVECTOR_H_

#include <iostream>
#include <math.h>
#include <comparator/moeoObjectiveObjectiveVectorComparator.h>

#include <comparator/moeoFuzzyParetoComparator.h>
//#include "triple.h"

/*
 *This class allows to represent a solution in the objective space (phenotypic representation) by a std::vector of triangles,
 * i.e. that an objective value is represented using a triangle of double, and this for any objective.
 */
template < class ObjectiveVectorTraits >
class moeoFuzzyRealObjectiveVector : public moeoObjectiveVector < ObjectiveVectorTraits, std::triple<double, double, double> >
  {
  public:

    using moeoObjectiveVector < ObjectiveVectorTraits, std::triple<double,double,double> >::size;
    using moeoObjectiveVector < ObjectiveVectorTraits, std::triple<double,double,double> >::operator[];

    /**
     * Ctor
     */
    moeoFuzzyRealObjectiveVector()
    {}


    /**
     * Ctor from a vector of triangles of doubles
     * @param _v the std::vector < std::triple<double, double, double> >
     */
   moeoFuzzyRealObjectiveVector(std::vector < std::triple<double,double,double> > & _v) : moeoObjectiveVector < ObjectiveVectorTraits, std::triple<double,double,double> > (_v)
    {}


    /**
     * Returns true if the current objective vector dominates _other according to the Fuzzy Preto dominance relation
     * (but it's better to use a moeoObjectiveVectorComparator object to compare solutions)
     * @param _other the other FuzzyRealObjectiveVector object to compare with
     */
    bool dominates(const moeoFuzzyRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        moeoFuzzyParetoComparator < moeoFuzzyRealObjectiveVector<ObjectiveVectorTraits> > comparator;
        return comparator(_other, *this);
      }


    /**
     * Returns true if the current objective vector is equal to _other (according to a tolerance value)
     * @param _other the other FuzzyRealObjectiveVector object to compare with
     */
   bool operator==(const moeoFuzzyRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        for (unsigned int i=0; i < size(); i++)
          {
            if ( (fabs((operator[](i)).first - _other[i].first) > ObjectiveVectorTraits::tolerance()) && 
                 (fabs((operator[](i)).second - _other[i].second) > ObjectiveVectorTraits::tolerance())  &&
                 (fabs((operator[](i)).third - _other[i].third) > ObjectiveVectorTraits::tolerance()) )
              {
                return false;
              }
          }
        return true;
      }
  
    /**
     * Returns true if the current objective vector is different than _other (according to a tolerance value)
     * @param _other the other FuzzyRealObjectiveVector object to compare with 
     */
    bool operator!=(const moeoFuzzyRealObjectiveVector < ObjectiveVectorTraits > & _other) const
      {
        return ! operator==(_other);
      }

};

/**
 * Output for a FuzzyRealObjectiveVector object
 * @param _os output stream
 * @param _objectiveVector the objective vector to write
 */

template < class ObjectiveVectorTraits >
std::ostream & operator<<(std::ostream & _os, const moeoFuzzyRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
  for (unsigned int i=0; i<_objectiveVector.size()-1; i++)
      _os << "[" << _objectiveVector[i].first << " " << _objectiveVector[i].second << " " << _objectiveVector[i].third << "]" <<" ";
      _os << "[" <<_objectiveVector[_objectiveVector.size()-1].first << " " << _objectiveVector[_objectiveVector.size()-1].second
          << " " << _objectiveVector[_objectiveVector.size()-1].third << "]" << " ";
  return _os;
}


/**
 * Input for a FuzzyRealObjectiveVector object
 * @param _is input stream
 * @param _objectiveVector the objective vector to read
 */
template < class ObjectiveVectorTraits >
std::istream & operator>>(std::istream & _is, moeoFuzzyRealObjectiveVector < ObjectiveVectorTraits > & _objectiveVector)
{
  
  for (unsigned int i=0; i<_objectiveVector.size(); i++)
    {
      _is >> _objectiveVector[i].first >> _objectiveVector[i].second >> _objectiveVector[i].third;
    }
  return _is;
}

#endif /*MOEOFUZZYREALOBJECTIVEVECTOR_H_*/
