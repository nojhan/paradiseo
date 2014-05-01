/*
* <moeoNearestNeighborDiversityAssignment.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Lille-Nord Europe, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Arnaud Liefooghe
* Jeremie Humeau
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
// moeoNearestNeighborDiversityAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEONEARESTNEIGHBORDIVERSITYASSIGNMENT_H_
#define MOEONEARESTNEIGHBORDIVERSITYASSIGNMENT_H_

#include <list>
#include <diversity/moeoDiversityAssignment.h>
#include <archive/moeoUnboundedArchive.h>
#include <archive/moeoArchive.h>

/**
 * moeoNearestNeighborDiversityAssignment is a moeoDiversityAssignment
 * using distance between individuals to assign diversity. Proposed in:
 * E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the
 * Strength Pareto Evolutionary Algorithm. Technical Report 103,
 * Computer Engineering and Networks Laboratory (TIK), ETH Zurich,
 * Zurich, Switzerland, 2001.

 * It is used in moeoSPEA2.
 */
template < class MOEOT >
class moeoNearestNeighborDiversityAssignment : public moeoDiversityAssignment < MOEOT >
{
public:

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor
     * @param _index index for find the k-ieme nearest neighbor, _index correspond to k
     */
    moeoNearestNeighborDiversityAssignment(unsigned int _index=1):distance(defaultDistance), archive(defaultArchive), index(_index)
    {}


    /**
     * Ctor where you can choose your own archive
     * @param _archive the archive used
     * @param _index index for find the k-ieme nearest neighbor, _index correspond to k
     */
    moeoNearestNeighborDiversityAssignment(moeoArchive <MOEOT>& _archive, unsigned int _index=1) : distance(defaultDistance), archive(_archive), index(_index)
    {}


    /**
     * Ctor where you can choose your own distance
     * @param _dist the distance used
     * @param _index index for find the k-ieme nearest neighbor, _index correspond to k
     */
    moeoNearestNeighborDiversityAssignment(moeoDistance <MOEOT, double>& _dist, unsigned int _index=1) : distance(_dist), archive(defaultArchive), index(_index)
    {}


    /**
     * Ctor where you can choose your own distance and archive
     * @param _dist the distance used
     * @param _archive the archive used
     * @param _index index for find the k-ieme nearest neighbor, _index correspond to k
     */
    moeoNearestNeighborDiversityAssignment(moeoDistance <MOEOT, double>& _dist, moeoArchive <MOEOT>& _archive, unsigned int _index=1) : distance(_dist), archive(_archive), index(_index)
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
     * @warning NOT IMPLEMENTED, DOES NOTHING !
     * Updates the diversity values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     * @warning NOT IMPLEMENTED, DOES NOTHING !
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        std::cout << "WARNING : updateByDeleting not implemented in moeoNearestNeighborDiversityAssignment" << std::endl;
    }


private:

    /** Distance */
    moeoDistance <MOEOT, double> & distance;
    /** Default distance */
    moeoEuclideanDistance < MOEOT > defaultDistance;
    /** Archive */
    moeoArchive < MOEOT > & archive;
    /** Default archive */
    moeoUnboundedArchive < MOEOT > defaultArchive;
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

#endif /*MOEONEARESTNEIGHBORDIVERSITYASSIGNEMENT_H_*/
