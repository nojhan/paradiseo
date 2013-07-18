/*
  <moSimpleMetropolisHastingsExplorer.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau, Lionel Parreaux

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.

  ParadisEO WebSite : http://paradiseo.gforge.inria.fr
  Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef _moSimpleMetropolisHastingsExplorer_h
#define _moSimpleMetropolisHastingsExplorer_h
/*
#include <cstdlib>

#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>

#include <utils/eoRNG.h>*/
#include <explorer/moMetropolisHastingsExplorer.h>


/**
 * Explorer for the Metropolis-Hasting Sampling.
 * Only the symetric case is considered when Q(x,y) = Q(y,x)
 * Fitness must be > 0
 */
template< class Neighbor >
class moSimpleMetropolisHastingsExplorer
: public moMetropolisHastingsExplorer< Neighbor, moSimpleMetropolisHastingsExplorer<Neighbor> >
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;
    
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;
    
    /**
     * Constructor for the simple MH explorer
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _maxSteps maximum number of currentStepNb to do
     * @param _solNeighborComparator a solution vs neighbor comparator
     */
    moSimpleMetropolisHastingsExplorer (
        Neighborhood& _neighborhood,
        moEval<Neighbor>& _eval,
        unsigned int _maxSteps,
        eoOptional< moSolNeighborComparator<Neighbor> > _comp = NULL
    )
    : moMetropolisHastingsExplorer< Neighbor, moSimpleMetropolisHastingsExplorer<Neighbor> >(_neighborhood, _eval, _comp)
    , maxSteps(_maxSteps)
    { }
    
    /**
     * initialization of currentStepNb to be done here
     * @param _solution unused
     */
    virtual void initParam(EOT & _solution) {
        currentStepNb = 0;
        //isAccept = true;
    };
    
    /**
     * increment currentStepNb
     * @param _solution unused
     */
    virtual void updateParam(EOT & _solution) {
        currentStepNb++;
    };
    
    /**
     * continue if there is a neighbor and it is remainds some steps to do
     * @param _solution the solution
     * @return true if there is still some steps to perform
     */
    virtual bool isContinue(EOT & _solution) {
        return currentStepNb < maxSteps;
    };
    
    /**
     * alpha required by moMetropolisHastingsExplorer
     * @param _solution the solution
     * @return a real between 0 and 1 representing the probability of accepting a worse solution
     */
    double alpha(EOT & _solution) {
        if (selectedNeighbor.fitness() == 0)
            return selectedNeighbor.fitness() < (double) _solution.fitness() ? 1: 0;
        return (double) _solution.fitness() / (double) selectedNeighbor.fitness();
    }
    
private:
    
    // current number of steps
    unsigned int currentStepNb;
    
    // maximum number of steps to perform
    unsigned int maxSteps;
    
};


#endif
