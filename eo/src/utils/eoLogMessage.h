/*
(c) Thales group, 2020

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

#ifndef _eoLogMessage_h
#define _eoLogMessage_h

#include <string>
#include "eoUpdater.h"
#include "eoLogger.h"

/**
   An updater that print its message when called (usually within an eoCheckPoint)

    @ingroup Utilities
*/
class eoLogMessage : public eoUpdater
{
public :
    eoLogMessage(
            std::string msg,
            eoLogger& log = eo::log,
            eo::Levels level = eo::progress
        ) :
            _msg(msg),
            _log(log),
            _level(level)
    { }

    virtual void operator()()
    {
        _log << _level << _msg << std::endl;
    }

protected:
    std::string _msg;
    eoLogger& _log;
    eo::Levels _level;

};

#endif
