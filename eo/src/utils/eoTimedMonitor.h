// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTimedMonitor.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2005
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mkeijzer@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _eoTimedMonitor_h
#define _eoTimedMonitor_h

#include <ctime>
#include <string>

#include <utils/eoMonitor.h>
#include <eoObject.h>

/**
    Holds a collection of monitors and only fires them when a time limit
    has been reached

    @ingroup Monitors
*/
class eoTimedMonitor : public eoMonitor
{
public:

    /** Constructor

    No negative time can be specified, use 0 if you want it to fire "always".
    @param seconds_ Specify time limit (s).
    */
    eoTimedMonitor(unsigned seconds_) : last_tick(0), seconds(seconds_) {}

    eoMonitor& operator()(void) {
        bool monitor = false;
        if (last_tick == 0) {
            monitor = true;
            last_tick = clock();
        }

        clock_t tick = clock();

        if ( (unsigned)(tick-last_tick) >= seconds * CLOCKS_PER_SEC) {
            monitor = true;
        }

        if (monitor) {
            for (unsigned i = 0; i < monitors.size(); ++i) {
                (*monitors[i])();
            }
            last_tick = clock();
        }
        return *this;
    }

    void add(eoMonitor& mon) { monitors.push_back(&mon); }

  virtual std::string className(void) const { return "eoTimedMonitor"; }

private:

    clock_t last_tick;

    unsigned seconds;

    std::vector<eoMonitor*> monitors;
};

#endif
