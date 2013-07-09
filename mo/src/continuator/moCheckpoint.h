/*
  <moCheckpoint.h>
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

#ifndef moCheckpoint_h
#define moCheckpoint_h

#include <continuator/moContinuator.h>
#include <utils/eoMonitor.h>
#include <continuator/moStatBase.h>
#include <utils/eoUpdater.h>
#include <continuator/moUpdater.h>
#include <neighborhood/moNeighborhood.h>

/**
 * Continuator allowing to add others (continuators, stats, monitors or updaters)
 */
template <class Neighbor>
class moCheckpoint : public moContinuator<Neighbor> {
public :

    typedef typename Neighbor::EOT EOT ;

    /**
     * Constructor (moCheckpoint must have at least one continuator)
     * @param _cont a continuator
     * @param _interval frequency to compute statistical operators
     */
    moCheckpoint(moContinuator<Neighbor>& _cont, unsigned int _interval=1):interval(_interval), counter(0) {
      continuators.push_back(&_cont);
    }

    /**
     * add a continuator to the checkpoint
     * @param _cont a continuator
     */
    void add(moContinuator<Neighbor>& _cont) {
        continuators.push_back(&_cont);
    }

    /**
     * add a statistic operator to the checkpoint
     * @param _stat a statistic operator
     */
    void add(moStatBase<EOT>& _stat) {
        stats.push_back(&_stat);
    }

    /**
     * add a monitor to the checkpoint
     * @param _mon a monitor
     */
    void add(eoMonitor& _mon) {
        monitors.push_back(&_mon);
    }

    /**
     * add a updater to the checkpoint
     * @param _upd an updater
     */
    void add(eoUpdater& _upd) {
        updaters.push_back(&_upd);
    }

    /**
     * add a MO updater to the checkpoint
     * @param _moupd an mo updater
     */
    void add(moUpdater& _moupd) {
        moupdaters.push_back(&_moupd);
    }

    /**
     * init all continuators containing in the checkpoint regarding a solution
     * @param _sol the corresponding solution
     */
    virtual void init(EOT& _sol) {
        for (unsigned i = 0; i < stats.size(); ++i)
            stats[i]->init(_sol);
        counter = 1;
        
        //for (unsigned i = 0; i < updaters.size(); ++i)
        //    updaters[i]->init();
        
        for (unsigned i = 0; i < moupdaters.size(); ++i)
            moupdaters[i]->init();
        /*
         * Removed because there was no reason for it to be done here.
         * It caused premature monitoring of eoParams with undefined values
         * 
        for (unsigned int i = 0; i < monitors.size(); ++i)
            (*monitors[i])();
        */
        
        for (unsigned i = 0; i < continuators.size(); ++i)
            continuators[i]->init(_sol);
    }

    /**
     * @return class name
     */
    virtual std::string className(void) const {
        return "moCheckpoint";
    }

    /**
     * apply operator of checkpoint's containers
     * @param _sol reference of the solution
     * @return true if all continuator return true
     */
    bool operator()(EOT & _sol) {
        unsigned i;
        bool bContinue = true;

        for (i = 0; i < stats.size(); ++i) {
            if (counter % interval == 0)
                (*stats[i])(_sol);
            counter++;
        }

        for (i = 0; i < updaters.size(); ++i)
            (*updaters[i])();

        for (i = 0; i < moupdaters.size(); ++i)
            (*moupdaters[i])();

        for (i = 0; i < monitors.size(); ++i)
            (*monitors[i])();

        for (i = 0; i < continuators.size(); ++i)
            if ( !(*continuators[i])(_sol) )
                bContinue = false;

        return bContinue;
    }

    /**
     * last call of statistic operators, monitors and updaters
     * @param _sol reference of the solution
     */
    void lastCall(EOT& _sol) {
        unsigned int i;
        for (i = 0; i < stats.size(); ++i)
            stats[i]->lastCall(_sol);

        for (i = 0; i < updaters.size(); ++i)
            updaters[i]->lastCall();

        for (i = 0; i < moupdaters.size(); ++i)
            moupdaters[i]->lastCall();

        for (i = 0; i < monitors.size(); ++i)
            monitors[i]->lastCall();
    }

private :
    /** continuators vector */
    std::vector<moContinuator<Neighbor>*> continuators;
    /** statistic operators vector */
    std::vector<moStatBase<EOT>*> stats;
    /** monitors vector */
    std::vector<eoMonitor*> monitors;
    /** updaters vector */
    std::vector<eoUpdater*> updaters;
    /** MO updaters vector */
    std::vector<moUpdater*> moupdaters;

    unsigned int interval;
    unsigned int counter;

};


#endif
