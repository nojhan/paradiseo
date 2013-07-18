/*
  <moMetropolisHastingsExplorer.h>
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

#ifndef _moMetropolisHastingsExplorer_h
#define _moMetropolisHastingsExplorer_h

#include <cstdlib>

#include <eoOptional.h>
#include <explorer/moNeighborhoodExplorer.h>
#include <comparator/moNeighborComparator.h>
#include <comparator/moSolNeighborComparator.h>
#include <neighborhood/moNeighborhood.h>

#include <utils/eoLogger.h>
#include <utils/eoRNG.h>

/**
 * Explorer for the Metropolis-Hasting Sampling.
 * Only the symetric case is considered when Q(x,y) = Q(y,x)
 * Fitness must be > 0
 * 
 * Class Derived passed in template parameter should define an "alpha" function
 * returning a real between 0 and 1 representing the probability of accepting
 * a worse solution
 */
template< class Neighbor, class Derived >
class moMetropolisHastingsExplorer : public moNeighborhoodExplorer<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moNeighborhoodExplorer<Neighbor>::eval;
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;

    /**
     * Constructor
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _comp a neighbor comparator
     */
    moMetropolisHastingsExplorer(
        Neighborhood& _neighborhood,
        moEval<Neighbor>& _eval,
        eoOptional< moSolNeighborComparator<Neighbor> > _comp = NULL
    )
    : moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval)
    , defaultSolNeighborComp(NULL)
    , solNeighborComparator(_comp.hasValue()? _comp.get(): *(defaultSolNeighborComp = new moSolNeighborComparator<Neighbor>()))
    {
        if (!neighborhood.isRandom()) {
            eo::log << eo::warnings << "moMetropolisHastingsExplorer::Warning -> the neighborhood used is not random" << std::endl;
        }
    }

    /**
     * Destructor
     */
    ~moMetropolisHastingsExplorer() {
        if (defaultSolNeighborComp != NULL)
            delete defaultSolNeighborComp;
    }
    
    /**
     * Init Search parameters: Nothing to do
     * @param _solution the solution to explore
     */
    virtual void initParam(EOT & _solution) {};
    
    /**
     * terminate: Nothing to do
     * @param _solution the solution to explore
     */
    virtual void terminate(EOT & _solution) {};
    
    /**
     * Explore the neighborhood of a solution
     * @param _solution
     */
    virtual void operator()(EOT & _solution)
    {
        //Test if _solution has a Neighbor
        if (neighborhood.hasNeighbor(_solution))
        {
            //init the first neighbor
            neighborhood.init(_solution, selectedNeighbor);
            
            //eval the _solution moved with the neighbor and stores the result in the neighbor
            eval(_solution, selectedNeighbor);
        }
    };
    
    /**
     * accept chooses whether to accept or reject the selected neighbor
     * @param _solution the solution
     * @return true if the selected neighbor is accepted
     */
    virtual bool accept(EOT & _solution)
    {
        bool isMoveAccepted = false;
        if (neighborhood.hasNeighbor(_solution)) {
            if (solNeighborComparator(_solution, selectedNeighbor))
                // accept if the current neighbor is better than the solution
                 isMoveAccepted = true;
            else isMoveAccepted = rng.uniform() < static_cast<Derived*>(this)->alpha(_solution);
        }
        return isMoveAccepted;
    };
    
private:
    
    // default comparator betwenn solution and neighbor
    moSolNeighborComparator<Neighbor>* defaultSolNeighborComp;
    
    // comparator betwenn solution and neighbor
    moSolNeighborComparator<Neighbor>& solNeighborComparator;
    
};


#endif
