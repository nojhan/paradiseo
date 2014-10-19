/*
  <moCounterMonitorSaver.h>
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

#ifndef moCounterMonitorSaver_h
#define moCounterMonitorSaver_h

#include "../../eo/utils/eoUpdater.h"
#include "../../eo/utils/eoMonitor.h"

/**
 * Class calling monitors with a given frequency
 */
class moCounterMonitorSaver : public eoUpdater {
public :

    /**
     * Constructor
     * @param _interval frequency to call monitors
     * @param _mon a monitor
     */
    moCounterMonitorSaver(unsigned _interval, eoMonitor& _mon) : interval(_interval), counter(0) {
        monitors.push_back(&_mon);
    }

    /**
     * call monitors if interval is reach by a counter
     */
    void operator()(void) {
        if (counter++ % interval == 0)
            for (unsigned i = 0; i < monitors.size(); i++)
                (*monitors[i])();
    }

    /**
     * last call of monitors
     */
    void lastCall(void) {
        for (unsigned i = 0; i < monitors.size(); i++)
            monitors[i]->lastCall();
    }

    /**
     * attach another monitor to this class
     * @param _mon the monitor to attach
     */
    void add(eoMonitor& _mon) {
        monitors.push_back(&_mon);
    }

    /**
     * @return name of the class
     */
    virtual std::string className(void) const {
        return "moCounterMonitorSaver";
    }

private :
    /** interval and counter value */
    unsigned int interval, counter;

    /** monitor's vector */
    std::vector<eoMonitor*> monitors;
};


#endif
