/*
 <moLocalSearch.h>
 Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

 Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moLocalSearch_h
#define _moLocalSearch_h

#include <explorer/moNeighborhoodExplorer.h>
#include <continuator/moContinuator.h>
#include <neighborhood/moNeighborhood.h>
#include <eoEvalFunc.h>
#include <eoOp.h>

/**
 * the main algorithm of the local search
 */
template<class Neighbor>
class moLocalSearch: public eoMonOp<typename Neighbor::EOT> {
public:
    typedef moNeighborhood<Neighbor> Neighborhood;
    typedef moNeighborhoodExplorer<Neighbor> NeighborhoodExplorer;
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor of a moLocalSearch
     * @param _searchExpl a neighborhood explorer
     * @param _cont an external continuator (can be a checkpoint!)
     * @param _fullEval a full evaluation function
     */
    moLocalSearch(NeighborhoodExplorer& _searchExpl, moContinuator<Neighbor> & _cont, eoEvalFunc<EOT>& _fullEval)
    : searchExplorer(_searchExpl), cont(&_cont), fullEval(_fullEval), currentSolutionFitness(0)
    { }

    /**
     * Run the local search on a solution
     * @param _solution the related solution
     */
    virtual bool operator()(EOT & _solution) {

        if (_solution.invalid())
            fullEval(_solution);
        
        // initialization of the parameter of the search (for example fill empty the tabu list)
        searchExplorer.initParam(_solution);
        
        // initialization of the external continuator (for example the time, or the number of generations)
        cont->init(_solution);
        
        bool b;
        do {
            currentSolutionFitness = _solution.fitness();
            //std::cout << currentSolutionFitness << std::endl;
            //std::cin.get();
            
            // explore the neighborhood of the solution
            searchExplorer(_solution);
            // if a solution in the neighborhood can be accepted
            if (searchExplorer.accept(_solution)) {
                searchExplorer.move(_solution);
                searchExplorer.moveApplied(true);
            } else
                searchExplorer.moveApplied(false);

            // update the parameter of the search (for ex. Temperature of the SA)
            searchExplorer.updateParam(_solution);

            b = (*cont)(_solution);
        } while (b && searchExplorer.isContinue(_solution));

        searchExplorer.terminate(_solution);

        cont->lastCall(_solution);

        return true;
    }

    /**
     * Set an external continuator
     * @param _cont the external continuator
     */
    void setContinuator(moContinuator<Neighbor> & _cont) {
        cont = &_cont;
    }

    /**
     * external continuator object
     *
     * @overload
     * @return the external continuator
     */
    moContinuator<Neighbor>* getContinuator() const {
        return cont;
    }

    /**
     * to get the neighborhood explorer
     *
     * @overload
     * @return the neighborhood explorer
     */
    moNeighborhoodExplorer<Neighbor> & getNeighborhoodExplorer() const {
        return searchExplorer;
    }
    
    // TODO doc
    double getCurrentSolutionFitness() const {
        return currentSolutionFitness;
    }

protected:
    // make the exploration of the neighborhood according to a local search heuristic
    moNeighborhoodExplorer<Neighbor>& searchExplorer;

    // external continuator
    moContinuator<Neighbor> * cont;

    //full evaluation function
    eoEvalFunc<EOT>& fullEval;
    
private:
    double currentSolutionFitness;
    
};

#endif

