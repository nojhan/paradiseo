/*
  <moBestSoFarStat.h>
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

#ifndef moBestSoFarStat_h
#define moBestSoFarStat_h

#include "moStat.h"

/**
 * The statistic which save the best solution found during the search
 */
template <class EOT>
class moBestSoFarStat : public moStat<EOT, EOT>
{
public :
    using moStat< EOT , EOT >::value;

    /**
     * Default Constructor
     * @param _reInitSol when true the best so far is reinitialized
     */
    moBestSoFarStat(bool _reInitSol = true): moStat<EOT, EOT>(EOT(), "best"), reInitSol(_reInitSol), firstTime(true) {
    }

    /**
     * Initialization of the best solution on the first one
     * @param _sol the first solution
     */
    virtual void init(EOT & _sol) {
    	if (reInitSol)
    		value() = _sol;
    	else if (firstTime)
			{
    			value() = _sol;
    			firstTime = false;
    		}
    }

    /**
     * Update the best solution
     * @param _sol the current solution
     */
    virtual void operator()(EOT & _sol) {
        if (value().fitness() < _sol.fitness())
        {
            value() = _sol;
        }
    }

    /**
     * @return name of the class
     */
    virtual std::string className(void) const {
        return "moBestSoFarStat";
    }

protected:
    bool reInitSol;
    bool firstTime;

};

#endif
