/*
  <moDynSpanCoolingSchedule.h>
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

#ifndef _moDynSpanCoolingSchedule_h
#define _moDynSpanCoolingSchedule_h

#include <coolingSchedule/moCoolingSchedule.h>

/**
 * Cooling Schedule of the temperature in the simulated algorithm
 * dynamic span : maximum number of tries and maximum number of moves
 * stop on the number of span with no move 
 *
 */
template< class EOT >
class moDynSpanCoolingSchedule : public moCoolingSchedule<EOT>
{
public:
    /**
     * default constructor
     * @param _initT initial temperature
     * @param _alpha factor of decreasing
     * @param _spanTriesMax maximum number of total move at equal temperature
     * @param _spanMoveMax maximum number of moves at equal temperature
     * @param _nbSpanMax maximum number of span with no move before stopping the search
     */
    moDynSpanCoolingSchedule(double _initT, double _alpha, unsigned int _spanTriesMax, unsigned int _spanMoveMax, unsigned int _nbSpanMax) : 
      initT(_initT), alpha(_alpha), spanTriesMax(_spanTriesMax), spanMoveMax(_spanMoveMax), nbSpanMax(_nbSpanMax) {
    }

    /**
     * Initial temperature
     * @param _solution initial solution
     */
    virtual double init(EOT & _solution) {
        // number of tries since the last temperature change
        spanTries = 0;

        // number of move since the last temperature change
        spanMove = 0;

        // number of successive span with no move
        nbSpan = 0;

        return initT;
    }

  /**
   * update the temperature by a factor
   * @param _temp current temperature to update
   * @param _acceptedMove true when the move is accepted, false otherwise
   * @param _currentSolution the current solution
   */
	virtual void update(double& _temp, bool _acceptedMove, EOT & _currentSolution) {
		spanTries++;
		
		if (_acceptedMove) 
			spanMove++;
		
		if (spanTries >= spanTriesMax || spanMove >= spanMoveMax) {
			_temp *= alpha;
			
			if (spanMove == 0) // no move during this span ?
				nbSpan++;
			else
				nbSpan = 0;
			
			spanTries = 0;
			spanMove = 0;
		}
	}
	
    /**
     * compare the number of span with no move
     * @param _temp current temperature
     * @return true if the search can continue
     */
    virtual bool operator()(double _temp) {
        return nbSpan <= nbSpanMax;
    }

private:
    // initial temperature
    double initT;

    // coefficient of decrease
    double alpha;

    // number of total move at equal temperature
    unsigned int spanTries;

    // number of move at equal temperature
    unsigned int spanMove;

    // number of successive spans with no move
    unsigned int nbSpan;

    // maximum number of total move at equal temperature
    unsigned int spanTriesMax;

    // maximum number of move at equal temperature
    unsigned int spanMoveMax;

    // maximum number of successive spans with no move
    unsigned int nbSpanMax;

};


#endif
