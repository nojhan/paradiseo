/*
* <moeoAggregativeFitnessAssignment.h>
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
// moeoAggregativeFitnessAssignment.h
//-----------------------------------------------------------------------------
#ifndef MOEOAGGREGATIVEFITNESSASSIGNMENT_H_
#define MOEOAGGREGATIVEFITNESSASSIGNMENT_H_

#include <vector>
#include <eoPop.h>

/*
 * Fitness assignment scheme which used weight foreach objectives
 */
template < class MOEOT >
class moeoAggregativeFitnessAssignment : public moeoFitnessAssignment < MOEOT >
{
public:


    /** the objective vector type of the solutions */
    typedef typename MOEOT::ObjectiveVector ObjectiveVector;


    /**
     * Default ctor
     * @param _weight vectors contains all weights.
     */
    moeoAggregativeFitnessAssignment(std::vector<double> & _weight) : weight(_weight)
    {}

    /**
     * Sets the fitness values for every solution contained in the population _pop  (and in the archive)
     * @param _pop the population
     */
    void operator()(eoPop < MOEOT > & _pop)
    {
    	unsigned int i= _pop.size();
        unsigned int nb_obj= MOEOT::ObjectiveVector::nObjectives();
        double res;        
        for (unsigned int k=0; k<i; k++){
        	res=0;
        	for(unsigned int l=0; l<nb_obj; l++)
        		res+=_pop[k].objectiveVector()[l] * weight[l];
        	_pop[k].fitness(res);
        }
    }


    /**
     * Warning: no yet implemented: Updates the fitness values of the whole population _pop by taking the deletion of the objective vector _objVec into account.
     * @param _pop the population
     * @param _objVec the objective vector
     */
    void updateByDeleting(eoPop < MOEOT > & _pop, ObjectiveVector & _objVec)
    {
        std::cout << "WARNING : updateByDeleting not implemented in moeoAssignmentFitnessAssignment" << std::endl;
    }


private:

	//the vector of weight
	std::vector<double> weight;

};

#endif /*MOEOAGGREGATIVEFITNESSASSIGNMENT_H_*/
