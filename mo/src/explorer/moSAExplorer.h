/*
  <moSAExplorer.h>
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

#ifndef _moSAExplorer_h
#define _moSAExplorer_h

#include <cstdlib>

//#include <explorer/moNeighborhoodExplorer.h>
#include <explorer/moMetropolisHastingsExplorer.h>
#include <comparator/moSolNeighborComparator.h>
#include <coolingSchedule/moCoolingSchedule.h>
#include <neighborhood/moNeighborhood.h>
#include <eoOptional.h>
#include <eval/moFullEvalByCopy.h>

#include <utils/eoRNG.h>

/**
 * Explorer for the Simulated Annealing
 * Only the symetric case is considered when Q(x,y) = Q(y,x)
 * Fitness must be > 0
 *
 */
template< class Neighbor >
class moSAExplorer : public moMetropolisHastingsExplorer< Neighbor, moSAExplorer<Neighbor> >
{
public:
    typedef typename Neighbor::EOT EOT ;
    typedef moNeighborhood<Neighbor> Neighborhood ;
    
    using moNeighborhoodExplorer<Neighbor>::neighborhood;
    using moNeighborhoodExplorer<Neighbor>::eval;
    using moNeighborhoodExplorer<Neighbor>::selectedNeighbor;
    
    
    /*moSAExplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval)
    : moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval)
    { }*/
    
    moSAExplorer (
        Neighborhood& _neighborhood,
        //eoOptional< moEval<Neighbor> > _eval                  = NULL,
        moEval<Neighbor>& _eval,
        moCoolingSchedule<EOT>& _cool,
        eoOptional< moSolNeighborComparator<Neighbor> > _comp = NULL
    )
    : moMetropolisHastingsExplorer< Neighbor, moSAExplorer<Neighbor> >(_neighborhood, _eval, _comp),
      /*moNeighborhoodExplorer<Neighbor>(_neighborhood, _eval.hasValue()? _eval.get(): *(default_eval = new moFullEvalByCopy<Neighbor>(_fullEval))),
      default_eval(NULL),             // removed in C++11 with unique_ptr*/
      //defaultSolNeighborComp(NULL),             // removed in C++11 with unique_ptr
      //solNeighborComparator(_comp.hasValue()? _comp.get(): *(defaultSolNeighborComp = new moSolNeighborComparator<Neighbor>())),
      //coolingSchedule(_coolingSchedule)
      coolingSchedule(_cool)
    {
        /*isMoveAccepted = false;
        if (!neighborhood.isRandom()) {
            std::cout << "moSAexplorer::Warning -> the neighborhood used is not random" << std::endl;
        }*/
    }

    /**
     * Destructor
     */
    ~moSAExplorer() {
    }

    /**
     * initialization of the initial temperature
     * @param _solution the solution
     */
    virtual void initParam(EOT & _solution) {
      temperature = coolingSchedule.init(_solution);
      //isMoveAccepted = false;
    };

    /**
     * decrease the temperature if necessary
     * @param _solution unused solution
     */
    virtual void updateParam(EOT & _solution) {
        coolingSchedule.update(temperature, this->moveApplied(), _solution);
    };

    /**
     * terminate: NOTHING TO DO
     * @param _solution unused solution
     */
    virtual void terminate(EOT & _solution) {};

    /**
     * Explore one random solution in the neighborhood
     * @param _solution the solution
     */
    virtual void operator()(EOT & _solution) {
        //Test if _solution has a Neighbor
        if (neighborhood.hasNeighbor(_solution)) {
            //init on the first neighbor: supposed to be random solution in the neighborhood
            neighborhood.init(_solution, selectedNeighbor);

            //eval the _solution moved with the neighbor and stock the result in the neighbor
            eval(_solution, selectedNeighbor);
        }
        else {
            //if _solution hasn't neighbor,
            //isMoveAccepted = false;
        }
    };

    /**
     * continue if the temperature is not too low
     * @param _solution the solution
     * @return true if the criteria from the cooling schedule is true
     */
    virtual bool isContinue(EOT & _solution) {
        return coolingSchedule(temperature);
    };

    /**
     * acceptance criterion according to the boltzmann criterion
     * @param _solution the solution
     * @return true if better neighbor or rnd < exp(delta f / T)
     */
//    virtual bool accept(EOT & _solution) {
//        if (neighborhood.hasNeighbor(_solution)) {
//            if (solNeighborComparator(_solution, selectedNeighbor)) // accept if the current neighbor is better than the solution
//                isMoveAccepted = true;
//            else {
//                /*
//                double alpha=0.0;
//                double fit1, fit2;
//                fit1=(double)selectedNeighbor.fitness();
//                fit2=(double)_solution.fitness();
//                if (fit1 < fit2) // this is a maximization
//                    alpha = exp((fit1 - fit2) / temperature );
//                else // this is a minimization
//                    alpha = exp((fit2 - fit1) / temperature );
//                isMoveAccepted = (rng.uniform() < alpha) ;*/
//                /*
//                double fit1 = (double)selectedNeighbor.fitness(),
//                       fit2 = (double)_solution.fitness(),
//                       alpha = fit1 < fit2 ? exp((fit1 - fit2) / temperature) : exp((fit2 - fit1) / temperature);
//                //if (fit1 < fit2) // this is a maximization
//                //else // this is a minimization
//                isMoveAccepted = (rng.uniform() < alpha);*/
//                /*double fit1 = (double) selectedNeighbor.fitness(),
//                       fit2 = (double) _solution.fitness(),
//                       alpha = fit1 < fit2 ? exp((fit1 - fit2) / temperature) : 1;
//                isMoveAccepted = (rng.uniform() <= alpha);*/
//                /*
//                double fit1 = (double) selectedNeighbor.fitness(),
//                       fit2 = (double) _solution.fitness(),
//                       alpha = exp( - fabs(fit1 - fit2) / temperature );
//                       // (fit1 - fit2) positive or negative depending on whether we're maximizing or minimizing
//                isMoveAccepted = (rng.uniform() <= alpha);*/
//                //isMoveAccepted = Accepter::accept(_solution, selectedNeighbor);
//                isMoveAccepted = static_cast<Derived*>(this)->accept(_solution);
//            }
//        }
//        return isMoveAccepted;
//    };

    /**
     * Getter
     * @return the temperature
     */
    double getTemperature() const {
        return temperature;
    }
    
    
    /*
    virtual bool doAccept(EOT & _solution) {
        double fit1 = (double) selectedNeighbor.fitness(),
               fit2 = (double) _solution.fitness(),
               alpha = exp( - fabs(fit1 - fit2) / temperature );
               // (fit1 - fit2) positive or negative depending on whether we're maximizing or minimizing
        return rng.uniform() <= alpha; 
    }*/
    double alpha(EOT & _solution) {
        //std::cout << "ok " << exp( - fabs((double) selectedNeighbor.fitness() - (double) _solution.fitness()) / temperature ) << " ";
        return exp( - fabs((double) selectedNeighbor.fitness() - (double) _solution.fitness()) / temperature );
    }
    
    
private:
    
    //moSolNeighborComparator<Neighbor>* defaultSolNeighborComp;
    
public://FIXME add friend
    // comparator betwenn solution and neighbor
    //moSolNeighborComparator<Neighbor>& solNeighborComparator;

    moCoolingSchedule<EOT>& coolingSchedule;

  // temperatur of the process
    double temperature;

    // true if the move is accepted
    //bool isMoveAccepted ;
    
    /**
     * Getter
     * @return the temperature
     */
    //virtual double getTemperature() = 0;
    
};


#endif

