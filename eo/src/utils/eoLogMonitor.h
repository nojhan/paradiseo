/*

(c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
(c) Thales group, 2010

    This library is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by the Free
    Software Foundation; version 2 of the license.

    This library is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along
    with this library; if not, write to the Free Software Foundation, Inc., 59
    Temple Place, Suite 330, Boston, MA 02111-1307 USA

Contact: http://eodev.sourceforge.net

Authors:
     Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoLogMonitor_h_
#define _eoLogMonitor_h_

#include <string>
#include <iostream>
#include <sstream>

#include "eoOStreamMonitor.h"
#include "eoLogger.h"

/**
    Prints statistics to a given ostream.

    You can pass any instance of an ostream to the constructor, like, for example, std::clog.

    @ingroup Monitors
*/
class eoLogMonitor : public eoOStreamMonitor
{
public :
    eoLogMonitor(
            eoLogger& logger = eo::log,
            eo::Levels level = eo::progress,
            std::string delim = "\t", unsigned int width=20, char fill=' ',
            bool print_names = false, std::string name_sep = ":"
        ) :
        eoOStreamMonitor(_oss, delim, width, fill, print_names, name_sep),
        _log(logger),
        _level(level)
    {}

    eoMonitor& operator()(void)
    {
        eoOStreamMonitor::operator()(); // write to _oss
        // Output at the log level.
        _log << _level << _oss.str();
        // Empty the output stream to avoid duplicated lines.
        _oss.str(""); _oss.clear();
        return *this;
    }

    virtual std::string className(void) const { return "eoLogMonitor"; }

private :
    std::ostringstream _oss;
    eoLogger & _log;
    eo::Levels _level;
};

#endif // _eoLogMonitor_h_
