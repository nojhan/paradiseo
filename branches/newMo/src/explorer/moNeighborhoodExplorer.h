/*
  <moNeighborhoodExplorer.h>
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

#ifndef _neighborhoodExplorer_h
#define _neighborhoodExplorer_h

//EO inclusion
#include <eoFunctor.h>

#include <neighborhood/moNeighborhood.h>
#include <eval/moEval.h>
#include <comparator/moNeighborComparator.h>

/**
 * Explore the neighborhood
 */
template< class NH >
class moNeighborhoodExplorer : public eoUF<typename NH::EOT&, void>
{
public:
    typedef NH Neighborhood ;
    typedef typename Neighborhood::EOT EOT ;
    typedef typename Neighborhood::Neighbor Neighbor ;

    /**
     * Constructor with a Neighborhood and evaluation function
     * @param _neighborhood the neighborhood
     * @param _eval the evaluation function
     * @param _comparator a neighbor comparator
     */
    moNeighborhoodExplorer(Neighborhood& _neighborhood, moEval<Neighbor>& _eval, moNeighborComparator<Neighbor>& _comparator):neighborhood(_neighborhood), eval(_eval), comparator(_comparator) {}

    /**
     * Init Search parameters
     * @param _solution the solution to explore
     */
    virtual void initParam (EOT& _solution) = 0 ;

    /**
     * Update Search parameters
     * @param _solution the solution to explore
     */
    virtual void updateParam (EOT& _solution) = 0 ;

    /**
     * Test if the exploration continue or not
     * @param _solution the solution to explore
     * @return true if the exploration continue, else return false
     */
    virtual bool isContinue(EOT& _solution) = 0 ;

    /**
     * Move a solution
     * @param _solution the solution to explore
     */
    virtual void move(EOT& _solution) = 0 ;

    /**
     * Test if a solution is accepted
     * @param _solution the solution to explore
     * @return true if the solution is accepted, else return false
     */
    virtual bool accept(EOT& _solution) = 0 ;

    /**
     * Terminate the search
     * @param _solution the solution to explore
     */
    virtual void terminate(EOT& _solution) = 0 ;

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
    	return "moNeighborhoodExplorer";
    }

protected:
    Neighborhood & neighborhood;
    moEval<Neighbor>& eval;
    moNeighborComparator<Neighbor>& comparator;
};

#endif
