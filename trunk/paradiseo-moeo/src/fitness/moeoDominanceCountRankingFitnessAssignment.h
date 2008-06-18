/*
* <moeoDominanceCountRankingFitnessAssignment.h>
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
// moeoDominanceCountRankingFitnessAssignment.h
//-----------------------------------------------------------------------------

#ifndef MOEODOMINANCECOUNTRANKINGFITNESSASSIGNMENT_H_
#define MOEODOMINANCECOUNTRANKINGFITNESSASSIGNMENT_H_

#include <vector>
#include <eoPop.h>
#include <archive/moeoArchive.h>
#include <archive/moeoUnboundedArchive.h>
#include <comparator/moeoObjectiveVectorComparator.h>
#include <comparator/moeoParetoObjectiveVectorComparator.h>
#include <fitness/moeoParetoBasedFitnessAssignment.h>
#include <utils/moeoDominanceMatrix.h>

/**
 * moeoDominanceCountRankingFitnessAssignment is a rank fitness assignment with value determine by a count fitness assignment.
 */
template < class MOEOT >
class moeoDominanceCountRankingFitnessAssignment : public moeoParetoBasedFitnessAssignment < MOEOT >
{
public:

    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;

    /**
     * Default ctor
     * @param _nocopy boolean to move away copies
     */
    moeoDominanceCountRankingFitnessAssignment(bool _nocopy=true) : comparator(paretoComparator), archive(defaultArchive), matrix(_nocopy)
    {}

    /**
     * Ctor where you can choose your own archive
     * @param _archive the archive used
     * @param _nocopy boolean to move away copies
     */
    moeoDominanceCountRankingFitnessAssignment(moeoArchive < MOEOT > & _archive, bool _nocopy=true) : comparator(paretoComparator), archive(_archive), matrix(_nocopy)
    {}

    /**
     * Ctor where you can choose your own way to compare objective vectors
     * @param _comparator the functor used to compare objective vectors
     * @param _nocopy boolean to move away copies
     */
    moeoDominanceCountRankingFitnessAssignment(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, bool _nocopy=true) : comparator(_comparator), archive(defaultArchive), matrix(_comparator, _nocopy)
    {}

    /**
     * Ctor where you can choose your own archive and your own way to compare objective vectors
     * @param _comparator the functor used to compare objective vectors
     * @param _archive the archive used
     * @param _nocopy boolean to move away copies
     */
    moeoDominanceCountRankingFitnessAssignment(moeoObjectiveVectorComparator < ObjectiveVector > & _comparator, moeoArchive < MOEOT > & _archive, bool _nocopy=true) : comparator(_comparator), archive(_archive), matrix(_comparator, _nocopy)
    {}


    /**
     * Sets the fitness values for every solution contained in the population _pop and in archive
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
        /*std::cout <<"\n\n";
        for(unsigned k=0;k<_pop.size();k++){
        	std::cout << "pop " << k << " :\n";
        	for(unsigned l=0 ; l<2 ; l++)
        		std::cout << "\tobjVec " << l << " : " << _pop[k].objectiveVector()[l] << "\n";
        }
        for(unsigned k=0;k<archive.size();k++){
        	std::cout << "archive " << k << " :\n";
        	for(unsigned l=0 ; l<2 ; l++)
        		std::cout << "\tobjVec " << l << " : " << archive[k].objectiveVector()[l] << "\n";
        }*/




        unsigned int i= _pop.size();
        unsigned int j= archive.size();
        matrix(archive,_pop);

        for (unsigned int k=0; k<j; k++)
            archive[k].fitness(countRanking(k));
        for (unsigned int k=j; k<i+j; k++)
        	_pop[k-j].fitness(countRanking(k));
    }

    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        //not yet implemented
    }


private:
    /** Functor to compare two objective vectors */
    moeoObjectiveVectorComparator < ObjectiveVector > & comparator;
    /** Functor to compare two objective vectors according to Pareto dominance relation */
    moeoParetoObjectiveVectorComparator < ObjectiveVector > paretoComparator;
    /** Archive */
    moeoArchive < MOEOT > & archive;
    /** Default archive */
    moeoUnboundedArchive < MOEOT > defaultArchive;
    /** Dominance Matrix*/
    moeoDominanceMatrix <MOEOT> matrix;

    double countRanking(unsigned int _i) {
        double res=0;
        for (unsigned int k=0; k<matrix.size(); k++) {
            if (matrix[k][_i])
                res+=matrix.count(k);
        }
        return -res;
    }

};

#endif /*MOEODOMINANCECOUNTRANKINGFITNESSASSIGNMENT_H_*/
