// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*
(c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000
(c) Thales group, 2010
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


Contact: http://eodev.sourceforge.net

Authors:
     todos@geneura.ugr.es
     Marc.Schoenauer@polytechnique.fr
     mkeijzer@dhi.dk
         Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoStdoutMonitor_h
#define _eoStdoutMonitor_h

#include <string>

#include <utils/eoOStreamMonitor.h>
#include <eoObject.h>

/**
    Prints statistics to stdout

    @ingroup Monitors
*/
class eoStdoutMonitor : public eoOStreamMonitor
{
public :
    /* FIXME remove in next release
    eoStdoutMonitor(bool _verbose, std::string _delim = "\t", unsigned int _width=20, char _fill=' ' ) :
       eoOStreamMonitor( std::cout, _verbose, _delim, _width, _fill)
    {
#ifndef DEPRECATED_MESSAGES
        eo::log << eo::warnings << "WARNING: the use of the verbose parameter in eoStdoutMonitor constructor is deprecated and will be removed in the next release" << std::endl;
#endif // !DEPRECATED_MESSAGES
    }
   */

    eoStdoutMonitor(std::string _delim = "\t", unsigned int _width=20, char _fill=' ' ) :
       eoOStreamMonitor( std::cout, _delim, _width, _fill)
    {}

    virtual std::string className(void) const { return "eoStdoutMonitor"; }
};

#endif
