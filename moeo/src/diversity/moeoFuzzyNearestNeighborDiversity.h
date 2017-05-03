/*
  <moeoFuzzyNearestNeighborDiversity.h>
   Oumayma BAHRI

Author:
       Oumayma BAHRI <oumaymabahri.com>

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr	   

*/
//-----------------------------------------------------------------------------

#ifndef MOEOFUZZYNEARESTNEIGHBORDIVERSITY_H_
#define MOEOFUZZYNEARESTNEIGHBORDIVERSITY_H_

#include <list>
#include <diversity/moeoDiversityAssignment.h>
#include <archive/moeoFuzzyArchive.h>
#include <distance/moeoBertDistance.h>

/**
 * moeoFuzzyNearestNeighborDiversity is a moeoDiversityAssignment using the fuzzy "Bert" distance between individuals to assign diversity.
 */
template < class MOEOT >
class moeoFuzzyNearestNeighborDiversity : public moeoDiversityAssignment < MOEOT >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Ctor where you can choose your own distance and archive
     * @param _dist the distance used
     * @param _archive the archive used
     * @param _index index for find the k-ieme nearest neighbor, _index correspond to k
     */
    moeoFuzzyNearestNeighborDiversity(moeoBertDistance <MOEOT, double>& _dist, moeoFuzzyArchive <MOEOT>& _archive, unsigned int _index=1) : distance(_dist), archive(_archive), index(_index)
    {}


    /**
     * Affect the diversity to the pop, diversity corresponding to the k-ieme nearest neighbor.
     * @param _pop the population
     */
    void operator () (eoPop < MOEOT > & _pop)
    {
        unsigned int i = _pop.size();
        unsigned int j = archive.size();
        double tmp=0;
        std::vector< std::list<double> > matrice(i+j);
        if (i+j>0)
        {
            for (unsigned k=0; k<i+j-1; k++)
            {
                for (unsigned l=k+1; l<i+j; l++)
                {
                    if ( (k<i) && (l<i) )
                        tmp=distance(_pop[k], _pop[l]);
                    else if ( (k<i) && (l>=i) )
                        tmp=distance(_pop[k], archive[l-i]);
                    else
                        tmp=distance(archive[k-i], archive[l-i]);
                    matrice[k].push_back(tmp);
                    matrice[l].push_back(tmp);
                }
            }
        }
        for (unsigned int k=0; k<i+j; k++)
            matrice[k].sort();
        for (unsigned int k=0; k<i; k++)
            _pop[k].diversity(-1 * 1/(2+getElement(matrice[k])));
        for (unsigned int k=i; k<i+j; k++)
            archive[k-i].diversity(-1 * 1/(2+getElement(matrice[k])));
    }


    /**
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        std::cout << "WARNING : updateByDeleting not implemented in moeoNearestNeighborDiversityAssignment" << std::endl;
    }


private:


    /** Default distance */
    moeoBertDistance < MOEOT >  Distance;
    /** Default archive */
    moeoFuzzyArchive < MOEOT > Archive;
    /** the index corresponding to k for search the k-ieme nearest neighbor */
    unsigned int index;


    /**
     * Return the index-th element of the list _myList
     * @param _myList the list which contains distances
     */
    double getElement(std::list<double> _myList)
    {
        std::list<double>::iterator it= _myList.begin();
        for (unsigned int i=1; i< std::min((unsigned int)_myList.size(),index); i++)
            it++;
        return *it;
    }

};

#endif /*MOEOFUZZYNEARESTNEIGHBORDIVERSITY_H_*/
