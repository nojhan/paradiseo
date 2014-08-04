/*
* <moeoDominanceMatrix.h>
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
// moeoDominanceMatrix.h
//-----------------------------------------------------------------------------
#ifndef MOEODOMINANCEMATRIX_H_
#define MOEODOMINANCEMATRIX_H_

#include <set>

/**
 * moeoDominanceMatrix allow to know if an MOEOT dominates another one or not. Can be apply on one or two eoPop.
 */
template <class MOEOT>
class moeoDominanceMatrix: public eoBF< eoPop< MOEOT >&, eoPop< MOEOT >& , void>,eoUF<eoPop <MOEOT>&,void>, std::vector < std::vector<bool> > {

public:

    using std::vector< std::vector<bool> >::size;
    using std::vector< std::vector<bool> >::resize;
    using std::vector< std::vector<bool> >::operator[];
    using std::vector< std::vector<bool> >::begin;
    using std::vector< std::vector<bool> >::end;

    /** The type for objective vector */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    /**
	 * Constructor which allow to choose the comparator
	 * @param _nocopy boolean allow to consider copy and doublons as bad element whose were dominated by all other MOEOT
	 * @param _comparator the comparator you want to use for the comparaison of two MOEOT
	 */
	moeoDominanceMatrix(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, bool _nocopy=true):std::vector < std::vector<bool> >(),comparator(_comparator), nocopy(_nocopy) {}

    /**
     * Default constructor with paretoComparator
     * @param _nocopy boolean allow to consider copy and doublons as bad element whose were dominated by all other MOEOT
     */
    moeoDominanceMatrix(bool _nocopy=false):std::vector < std::vector<bool> >(),comparator(paretoComparator), nocopy(_nocopy) {}



    /**
     * Filling up the Dominance Matrix on one population
     * @param _pop first population
     */
    void operator()(eoPop<MOEOT>& _pop) {
        eoPop <MOEOT> dummyPop;
        (*this).operator()(_pop, dummyPop);
    }

    /**
     * Filling up the Dominance Matrix of first and second population (if you have only one population, the second must be empty)
     * @param _pop1 first population
     * @param _pop2 second population
     */
    void operator()(eoPop<MOEOT>& _pop1,eoPop<MOEOT>& _pop2) {

        //Initialization
        unsigned int i= _pop1.size();
        unsigned int j= _pop2.size();
        resize(i+j);
        countVector.resize(i+j);
        rankVector.resize(i+j);
        for (unsigned int k=0; k < i+j; k++) {
            (*this)[k].resize(i+j);
            countVector[k]=0;
            rankVector[k]=0;
            for (unsigned l=0; l<i+j; l++)
                (*this)[k][l]=false;
        }
        //filling up of matrix, count and rank vectors
        for (unsigned int k=0; k<i+j-1; k++) {
            for (unsigned int l=k+1; l<i+j; l++) {
                if ( (k < i) && (l < i) ) {
                    if ( comparator(_pop1[l].objectiveVector(), _pop1[k].objectiveVector())) {
                        (*this)[k][l]=true;
                        countVector[k]++;
                        rankVector[l]++;
                    }
                    else if ( comparator(_pop1[k].objectiveVector(), _pop1[l].objectiveVector())) {
                        (*this)[l][k]=true;
                        countVector[l]++;
                        rankVector[k]++;
                    }
                    else if (nocopy && (_pop1[k].objectiveVector() == _pop1[l].objectiveVector()))
                        copySet.insert(l);
                }
                else if (( (k < i) && (l >= i) )) {
                    if ( comparator(_pop2[l-i].objectiveVector(), _pop1[k].objectiveVector())) {
                        (*this)[k][l]=true;
                        countVector[k]++;
                        rankVector[l]++;
                    }
                    else if ( comparator(_pop1[k].objectiveVector(), _pop2[l-i].objectiveVector())) {
                        (*this)[l][k]=true;
                        countVector[l]++;
                        rankVector[k]++;
                    }
                    else if (nocopy && (_pop1[k].objectiveVector() == _pop2[l-i].objectiveVector()))
                        copySet.insert(l);
                }
                else {
                    if ( comparator(_pop2[l-i].objectiveVector(), _pop2[k-i].objectiveVector())) {
                        (*this)[k][l]=true;
                        countVector[k]++;
                        rankVector[l]++;
                    }
                    else if ( comparator(_pop2[k-i].objectiveVector(), _pop2[l-i].objectiveVector())) {
                        (*this)[l][k]=true;
                        countVector[l]++;
                        rankVector[k]++;
                    }
                    else if (nocopy && (_pop2[k-i].objectiveVector() == _pop2[l-i].objectiveVector()))
                        copySet.insert(l);
                }
            }
        }


        //if we don't want copy, matrix, rankVector and countVector are updating
        if (nocopy) {
            std::set<unsigned int>::iterator it=copySet.begin();

            while (it!=copySet.end()) {
                for (unsigned int  l=0; l< (*this).size(); l++) {
                    if (!(*this)[l][*it]) {
                        (*this)[l][*it]=true;
                        countVector[l]++;
                    }
                }
                it++;
            }
            it=copySet.begin();
            while (it!=copySet.end()) {
                for (unsigned int  l=0; l< (*this).size(); l++) {
                    if ((*this)[*it][l]) {
                        (*this)[*it][l]=false;
                        rankVector[l]--;
                    }
                }
                it++;
            }
            it=copySet.begin();
            while (it!=copySet.end()) {
                countVector[*it]=0;
                rankVector[*it]=(*this).size()-copySet.size();
                it++;
            }
        }
        /*
        for(unsigned k=0; k<i+j; k++)
        	for(unsigned l=0; l<i+j; l++){
        		std::cout << (*this)[k][l];
        		if(l==i+j-1)
        			std::cout << "\n";
        		else
        			std::cout << "  ";
        	}

        for(unsigned k=0; k<i+j; k++){
        	std::cout << "rank " << k << " : " << rankVector[k] << "\n";
        	std::cout << "count " << k << " : " << countVector[k] << "\n";
        }*/

    }

    /**
     * @param _i the index of the element that we want RankDominanceFitness
     * @return RankDominanceFitness of element of index _i
     */
    double rank(unsigned int _i) {
        return rankVector[_i];
    }

    /**
     * @param _i the index of the element that we want CountDominanceFitness
     * @return CountDominanceFitness of element of index _i
     */
    double count(unsigned int _i) {
        return countVector[_i];
    }

private:
    /** Functor to compare two objective vectors */
    moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;
    /** boolean allow or not to pull away a copy*/
    bool nocopy;
    /** vector contains CountDominanceFitnessAssignment */
    std::vector<double> countVector;
    /** vector contains CountRankFitnessAssignment */
    std::vector<double> rankVector;
    /** vector contains index of copys */
    std::set<unsigned int> copySet;


};

#endif /*MOEODOMINANCEMATRIX_H_*/
