/*
  <moSimpleCoolingSchedule.h>
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

#ifndef _moSimpleCoolingSchedule_h
#define _moSimpleCoolingSchedule_h

#include <coolingSchedule/moCoolingSchedule.h>

/**
 * Classical cooling Schedule of the temperature in the simulated algorithm with initial and final temperature and a factor of decrease
 *
 */
template< class EOT >
class moSimpleCoolingSchedule : public moCoolingSchedule<EOT>
{
public:
    /**
     * Constructor
     * @param _initT initial temperature
     * @param _alpha factor of decreasing
     * @param _span number of iteration with equal temperature
     * @param _finalT final temperature, threshold of the stopping criteria
     */
    moSimpleCoolingSchedule(double _initT, double _alpha, unsigned _span, double _finalT) : initT(_initT), alpha(_alpha), span(_span), finalT(_finalT) {}

    /**
     * Getter on the initial temperature
     * @param _solution initial solution
     * @return the initial temperature
     */
    virtual double init(EOT & /*_solution*/) {
        // number of iteration with the same temperature
        step = 0;

        return initT;
    }

    /**
     * update the temperature by a factor
     * @param _temp current temperature to update
     * @param _acceptedMove true when the move is accepted, false otherwise
     */
    virtual void update(double& _temp, bool /*_acceptedMove*/) {
        if (step >= span) {
            _temp *= alpha;
            step = 0;
        } else
            step++;
    }

    /**
     * compare the temperature to the threshold
     * @param _temp current temperature
     * @return true if the current temperature is over the threshold (final temperature)
     */
    virtual bool operator()(double _temp) {
        return _temp > finalT;
    }

private:
    // initial temperature
    double initT;
    // coefficient of decrease
    double alpha;
    // maximum number of iterations at the same temperature
    unsigned int span;
    // threshold temperature
    double finalT;
    // number of steps with the same temperature
    unsigned int step;
};


#endif
