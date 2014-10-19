/*
* <moeoSPEA2Archive.h>
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
// moeoSPEA2Archive.h
//-----------------------------------------------------------------------------

#ifndef MOEOSPEA2ARCHIVE_H_
#define MOEOSPEA2ARCHIVE_H_

#include <limits>
#include <list>
#include "../../eo/eoPop.h"
#include "moeoFixedSizeArchive.h"
#include "../comparator/moeoComparator.h"
#include "../comparator/moeoFitnessThenDiversityComparator.h"
#include "../comparator/moeoObjectiveVectorComparator.h"
#include "../distance/moeoDistance.h"
#include "../distance/moeoEuclideanDistance.h"

/**
 * This class represents a bounded archive as defined in the SPEA2 algorithm.
 * E. Zitzler, M. Laumanns, and L. Thiele. SPEA2: Improving the Strength Pareto Evolutionary Algorithm. Technical Report 103,
 * Computer Engineering and Networks Laboratory (TIK), ETH Zurich, Zurich, Switzerland, 2001.
 */
template < class MOEOT >
class moeoSPEA2Archive : public moeoFixedSizeArchive < MOEOT >
{
public:

    using moeoFixedSizeArchive < MOEOT > :: size;
    using moeoFixedSizeArchive < MOEOT > :: resize;
    using moeoFixedSizeArchive < MOEOT > :: operator[];
    using moeoFixedSizeArchive < MOEOT > :: back;
    using moeoFixedSizeArchive < MOEOT > :: pop_back;
    using moeoFixedSizeArchive < MOEOT > :: push_back;
    using moeoFixedSizeArchive < MOEOT > :: begin;
    using moeoFixedSizeArchive < MOEOT > :: end;


    /**
     * The type of an objective vector for a solution
     */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor.
     * @param _maxSize the size of archive (must be smaller or equal to the population size)
     */
    moeoSPEA2Archive(unsigned int _maxSize=100): moeoFixedSizeArchive < MOEOT >(true), maxSize(_maxSize), borne(0), indiComparator(defaultComparator), distance(defaultDistance)
    {}


    /**l
     * Ctor where you can choose your own moeoDistance
     * @param _dist the distance used
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     */
    moeoSPEA2Archive(moeoDistance <MOEOT, double>& _dist, unsigned int _maxSize=100): moeoFixedSizeArchive < MOEOT >(true), maxSize(_maxSize), borne(0), indiComparator(defaultComparator), distance(_dist)
    {}


    /**
     * Ctor where you can choose your own moeoObjectiveVectorComparator
     * @param _comparator the functor used to compare objective vectors
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     */
    moeoSPEA2Archive(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, unsigned int _maxSize=100): moeoFixedSizeArchive < MOEOT >(_comparator, true), maxSize(_maxSize), borne(0), indiComparator(defaultComparator), distance(defaultDistance)
    {}


    /**
     * Ctor where you can choose your own moeoComparator
     * @param _indiComparator the functor used to compare MOEOT
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     */
    moeoSPEA2Archive(moeoComparator <MOEOT>& _indiComparator, unsigned int _maxSize=100): moeoFixedSizeArchive < MOEOT >(true), maxSize(_maxSize), borne(0), indiComparator(_indiComparator), distance(defaultDistance)
    {}


    /**
     * Ctor where you can choose your own moeoComparator, moeoDistance and moeoObjectiveVectorComparator
     * @param _indiComparator the functor used to compare MOEOT
     * @param _dist the distance used
     * @param _comparator the functor used to compare objective vectors
     * @param _maxSize the size of archive (must be smaller or egal to the population size)
     */
    moeoSPEA2Archive(moeoComparator <MOEOT>& _indiComparator, moeoDistance <MOEOT, double>& _dist, moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, unsigned int _maxSize=100) : moeoFixedSizeArchive < MOEOT >(_comparator, true), maxSize(_maxSize), borne(0), indiComparator(_indiComparator), distance(_dist)
    {}


    /**
     * Updates the archive with a given individual _moeo
     * @param _moeo the given individual
     * @return true (TODO)
     */
    bool operator()(const MOEOT & _moeo)
    {
        eoPop < MOEOT > pop_tmp;
        pop_tmp.push_back(_moeo);
        operator()(pop_tmp);
        return true;
    }


    /**
     * Updates the archive with a given population _pop
     * @param _pop the given population
     * @return true (TODO)
     */
    bool operator()(const eoPop < MOEOT > & _pop)
    {
        unsigned int i;
        unsigned int foo=0;

        //Creation of the vector that contains minimal pop's informations
        std::vector<struct refpop> copy_pop(_pop.size());
        for (i=0;i<_pop.size(); i++)
        {
            copy_pop[i].index=i;
            copy_pop[i].fitness=_pop[i].fitness();
            copy_pop[i].diversity=_pop[i].diversity();
        }

        //Sort this vector in decrease order of fitness+diversity
        std::sort(copy_pop.begin(), copy_pop.end(), Cmp());

        //If the archive is empty, put in the best elements of the pop
        if (borne < maxSize)
        {
            foo= std::min((unsigned int)_pop.size(), maxSize-borne);
            for (i=0; i< foo ; i++)
            {
                push_back(_pop[copy_pop[i].index]);
                borne++;
            }

        }
        else
        {
            unsigned int j=0;
            //Sort the archive
            std::sort(begin(), end(), indiComparator);
            i=0;

            //While we have a better element in pop than the worst <= -1 in the archive, replace the worst(of archive) by the best(of pop)
            while ( (i<borne) && ( (operator[](i).fitness()+operator[](i).diversity()) < (copy_pop[j].fitness + copy_pop[j].diversity) ) && (operator[](i).fitness()<=-1) && ( j < copy_pop.size() ) )
            {
                operator[](i)= back();
                pop_back();
                push_back(_pop[copy_pop[j].index]);
                i++;
                j++;
            }

            //If their are others goods elements in pop (fitness=0) , keep only archive's size elements between the archive's elements and the good element in the pop (k ieme smallest distance is used)
            if (copy_pop[j].fitness > -1)
            {
                unsigned int inf=j;
                unsigned int p;
                unsigned int k=0;
                unsigned int l=0;
                double tmp=0;
                unsigned int tmp2=0;

                //search bounds of copy_pop where are the goods elements
                while ((j < copy_pop.size()) && (copy_pop[j].fitness > -1.0))
                    j++;

                p=j-inf;

                std::vector< std::vector< std::pair<int,double> > > matrice(borne+p);

                //Build the distance matrice(vector of vector) between each keeped elements
                if (borne+p>0)
                {
                    for (k=0; k<borne+p-1; k++)
                    {
                        for (l=k+1; l<borne+p; l++)
                        {
                            if ( (k<borne) && (l<borne) )
                                tmp=distance(operator[](k), operator[](l));
                            else if ( (k<borne) && (l>=borne) )
                                tmp=distance(operator[](k), _pop[copy_pop[l-borne+inf].index]);
                            else
                                tmp=distance(_pop[copy_pop[k-borne+inf].index], _pop[copy_pop[l-borne+inf].index]);

                            matrice[k].push_back(std::pair<int,double>(l,tmp));
                            matrice[l].push_back(std::pair<int,double>(k,tmp));
                        }
                    }
                }

                for (k=0; k<borne+p; k++)
                {
                    //sort each line of the matrice
                    std::sort(matrice[k].begin(),matrice[k].end(), CmpPair());

                    //insert an indice at the end of each line after they were be sorted
                    matrice[k].push_back(std::pair<int,double>(-1,k));
                }

                //sort the lines of the matrice between us (by shortest distance)
                std::sort(matrice.begin(),matrice.end(), CmpVector());

                //vectors and iterators used to replace some archive element by some pop element
                std::vector<unsigned int> notkeeped;
                std::vector<unsigned int> keeped;
                std::vector< std::vector< std::pair<int,double> > >::iterator matrice_it=matrice.begin();
                std::vector< std::pair<int,double> >::iterator it;

                //search elements of the archive to delete
                for (k=0; k<p; k++)
                {
                    tmp2=(unsigned int)matrice[0].back().second;
                    if (tmp2<borne)
                        notkeeped.push_back(tmp2);
                    matrice.erase(matrice_it);
                    for (l=0; l<matrice.size(); l++)
                    {
                        it=matrice[l].begin();
                        while ((unsigned int)(*it).first != tmp2)
                            it++;
                        matrice[l].erase(it);
                    }
                    if (k != (p-1))
                        std::sort(matrice.begin(),matrice.end(), CmpVector());
                }

                //search elements of pop to put in archive
                for (k=0; k<borne; k++)
                {
                    tmp2=(unsigned int)matrice[k].back().second;
                    if (tmp2 >= borne)
                        keeped.push_back(tmp2);
                }

                //replace some archive element by some pop element
                for (k=0; k<keeped.size(); k++)
                {
                    push_back( _pop[ copy_pop[keeped[k]-borne+inf].index ] );
                    operator[](notkeeped[k]) = back();
                    pop_back();
                }
            }
        }
        return true;
    }//endoperator()


private:

    /** archive max size */
    unsigned int maxSize;
    /** archive size */
    unsigned int borne;
    /**
     * Wrapper which allow to used an moeoComparator in std::sort
     * @param _comp the comparator to used
     */
    class Wrapper
    {
    public:
        /**
         * Ctor.
         * @param _comp the comparator
         */
        Wrapper(moeoComparator < MOEOT > & _comp) : comp(_comp) {}
        /**
         * Returns true if _moeo1 is greater than _moeo2 according to the comparator
         * _moeo1 the first individual
         * _moeo2 the first individual
         */
        bool operator()(const MOEOT & _moeo1, const MOEOT & _moeo2)
        {
            return comp(_moeo1,_moeo2);
        }
    private:
        /** the comparator */
        moeoComparator < MOEOT > & comp;
    }
    indiComparator;
    /** default moeoComparator*/
    moeoFitnessThenDiversityComparator < MOEOT > defaultComparator;
    /** distance */
    moeoDistance <MOEOT, double>& distance;
    /** default distance */
    moeoEuclideanDistance < MOEOT > defaultDistance;


    /**
     * Structure needs to copy informations of the pop in order to sort it
     */
    struct refpop
    {
        unsigned int index;
        double fitness;
        double diversity;
    };


    /**
     * Comparator of struct refpop : compare fitness+divesity
     */
    struct Cmp
    {
        bool operator()(const struct refpop& _a, const struct refpop& _b)
        {
            return ( (_a.diversity + _a.fitness) > (_b.diversity + _b.fitness) );
        }
    };


    /**
     * Comparator of two vector of pair
     * Compare the second pair's value of the first element vector, if equals compare the next element vector...
     */
    struct CmpVector
    {
        bool operator()( const std::vector< std::pair<int,double> >& _a, const std::vector< std::pair<int,double> >& _b)
        {
            std::vector< std::pair<int,double> >::const_iterator it1= _a.begin();
            std::vector< std::pair<int,double> >::const_iterator it2= _b.begin();
            while ( (it1 != _a.end()) && (it2 != _b.end()))
            {
                if ((*it1).second < (*it2).second)
                    return true;
                else if ((*it1).second > (*it2).second)
                    return false;
                it1++;
                it2++;
            }
            return true;
        }
    };


    /**
       * Comparator of two pair : compare the second pair's value
       */
    struct CmpPair
    {
        bool operator()(const std::pair<int,double>& _a, const std::pair<int,double>& _b)
        {
            return _a.second < _b.second;
        }
    };


};

#endif /*MOEOSPEA2ARCHIVE_H_*/
