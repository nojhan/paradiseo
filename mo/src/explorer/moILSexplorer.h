/*
  <moILSexplorer.h>
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

#ifndef _moILSexplorer_h
#define _moILSexplorer_h

#include "moNeighborhoodExplorer.h"
#include "../comparator/moNeighborComparator.h"
#include "../comparator/moSolNeighborComparator.h"
#include "../algo/moLocalSearch.h"
#include "../perturb/moPerturbation.h"
#include "../acceptCrit/moAcceptanceCriterion.h"
//#include <neighborhood/moDummyNeighborhood.h>
//#include <neighborhood/moDummyNeighbor.h>

/**
 * Explorer for an Iterated Local Search
 */

//template< class Neighbor >
//class moILSexplorer : public moNeighborhoodExplorer< moDummyNeighbor<typename Neighbor::EOT> >

template< class Neighbor, class NeighborLO >
class moILSexplorer : public moNeighborhoodExplorer< NeighborLO >
{
public:
  typedef moNeighborhood<Neighbor> Neighborhood ;
  typedef typename Neighbor::EOT EOT;
  typedef moNeighborhood<NeighborLO> NeighborhoodLO;
  
  //typedef moDummyNeighbor<EOT> dummyNeighbor;
  //typedef moDummyNeighborhood<dummyNeighbor> dummyNeighborhood;

    /**
     * Constructor
     * @param _ls a local search
     * @param _perturb a perturbation operator
     * @param _acceptCrit a acceptance criteria
     */
    moILSexplorer(moLocalSearch<Neighbor>& _ls, moPerturbation<Neighbor>& _perturb, moAcceptanceCriterion<Neighbor>& _acceptCrit) : moNeighborhoodExplorer<NeighborLO>(), ls(_ls), perturb(_perturb), acceptCrit(_acceptCrit) {
        firstIteration=true;
    }

    /**
     * Destructor
     */
    ~moILSexplorer() {

    }

    /**
     * Init perturbation and acceptance criteria
     * @param _solution the current solution
     */
    virtual void initParam(EOT & _solution) {
        firstIteration=true;
        perturb.init(_solution);
        acceptCrit.init(_solution);
    };

    /**
     * Update perturbation and acceptance criteria
     * @param _solution the current solution
     */
    virtual void updateParam(EOT & _solution) {
        if ((*this).moveApplied()) {
            perturb.add(_solution,emptyNeighbor);
            acceptCrit.add(_solution,emptyNeighbor);
        }
        perturb.update(_solution, emptyNeighbor);
        acceptCrit.update(_solution, emptyNeighbor);
    };

    /**
     * terminate: NOTHING TO DO
     * @param _solution a solution (unused)
     */
    virtual void terminate(EOT & _solution) {};

    /**
     * Perturb and apply local search on a solution
     * @param _solution the solution
     */
    virtual void operator()(EOT & _solution) {
        //copy the solution to perform new local search
        currentSol=_solution;

        //perturb solution exept at the first iteration
        if (!firstIteration) {
            perturb(currentSol);
        }
        else
            firstIteration=false;

        //apply the local search on the copy
        ls(currentSol);

//        std::cout << "(solution)\t" << current << std::endl;

    };

    /**
     * Always continue
     * @param _solution the solution
     * @return always true
     */
    virtual bool isContinue(EOT & _solution) {
        return true;
    };

    /**
     * copy the solution found by the local search
     * @param _solution the solution
     */
    virtual void move(EOT & _solution) {
        _solution=currentSol;
    };

    /**
     * accept test if an ameliorated neighbor was found
     * @param _solution the solution
     * @return true if acceptance criteria is verified
     */
    virtual bool accept(EOT & _solution) {
        return acceptCrit(_solution, currentSol);
    };

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moILSexplorer";
    }

private:
    //Usefull to use the momory of tabuSearch
    Neighbor emptyNeighbor;
    EOT currentSol;
    moLocalSearch<Neighbor>& ls;
    moPerturbation<Neighbor> & perturb;
    moAcceptanceCriterion<Neighbor>& acceptCrit;

    bool firstIteration;
};


#endif
