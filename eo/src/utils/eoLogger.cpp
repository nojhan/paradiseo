// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the license.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Johann Dréo <johann.dreo@thalesgroup.com>
Caner Candan <caner.candan@thalesgroup.com>

*/

#ifdef _WIN32
#include <io.h>
#else // _WIN32
#include <unistd.h>
#endif // ! _WIN32

#include <fcntl.h>
#include <cstdlib>
#include <cstdio> // used to define EOF

#include <iostream>

#include "eoLogger.h"

void eoLogger::_init()
{
    _standard_io_streams[&std::cout] = 1;
    _standard_io_streams[&std::clog] = 2;
    _standard_io_streams[&std::cerr] = 2;

    // /!\ If you want to add a level dont forget to add it at the header file in the enumerator Levels

    addLevel("quiet", eo::quiet);
    addLevel("errors", eo::errors);
    addLevel("warnings", eo::warnings);
    addLevel("progress", eo::progress);
    addLevel("logging", eo::logging);
    addLevel("debug", eo::debug);
    addLevel("xdebug", eo::xdebug);
}

eoLogger::eoLogger() :
    std::ostream(&_obuf),

    _verbose("quiet", "verbose", "Set the verbose level", 'v'),
    _printVerboseLevels(false, "print-verbose-levels", "Print verbose levels", 'l'),
    _output("", "output", "Redirect a standard output to a file", 'o'),

    _selectedLevel(eo::progress),
    _contextLevel(eo::quiet),
    _fd(2),
    _obuf(_fd, _contextLevel, _selectedLevel)
{
    _init();
}

eoLogger::eoLogger(eo::file file) :
    std::ostream(&_obuf),

    _verbose("quiet", "verbose", "Set the verbose level", 'v'),
    _printVerboseLevels(false, "print-verbose-levels", "Print verbose levels", 'l'),
    _output("", "output", "Redirect a standard output to a file", 'o'),

    _selectedLevel(eo::progress),
    _contextLevel(eo::quiet),
    _fd(2),
    _obuf(_fd, _contextLevel, _selectedLevel)
{
    _init();
    *this << file;
}

eoLogger::~eoLogger()
{
    if (_fd > 2) { ::close(_fd); }
}

void eoLogger::_createParameters( eoParser& parser )
{
    //------------------------------------------------------------------
    // we are saying to eoParser to create the parameters created above.
    //------------------------------------------------------------------

    std::string section("Logger");
    parser.processParam(_verbose, section);
    parser.processParam(_printVerboseLevels, section);
    parser.processParam(_output, section);

    //------------------------------------------------------------------


    //------------------------------------------------------------------
    // we're gonna redirect the log to the given filename if -o is used.
    //------------------------------------------------------------------

    if ( ! _output.value().empty() )
        {
            eo::log << eo::file( _output.value() );
        }

    //------------------------------------------------------------------


    //------------------------------------------------------------------
    // we're gonna print the list of levels if -l parameter is used.
    //------------------------------------------------------------------

    if ( _printVerboseLevels.value() )
        {
            eo::log.printLevels();
        }

    //------------------------------------------------------------------
}

std::string eoLogger::className() const
{
    return ("eoLogger");
}

void eoLogger::addLevel(std::string name, eo::Levels level)
{
    _levels[name] = level;
    _sortedLevels.push_back(name);
}

void eoLogger::printLevels() const
{
    std::cout << "Available verbose levels:" << std::endl;

    for (std::vector<std::string>::const_iterator it = _sortedLevels.begin(), end = _sortedLevels.end();
         it != end; ++it)
        {
            std::cout << "\t" << *it << std::endl;
        }

    ::exit(0);
}

eoLogger& operator<<(eoLogger& l, const eo::Levels lvl)
{
    l._contextLevel = lvl;
    return l;
}

eoLogger& operator<<(eoLogger& l, eo::file f)
{
    l._fd = ::open(f._f.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    return l;
}

eoLogger& operator<<(eoLogger& l, eo::setlevel v)
{
    l._selectedLevel = (v._lvl < 0 ? l._levels[v._v] : v._lvl);
    return l;
}

eoLogger& operator<<(eoLogger& l, std::ostream& os)
{
    if (l._standard_io_streams.find(&os) != l._standard_io_streams.end())
        {
            l._fd = l._standard_io_streams[&os];
        }
    return l;
}

eoLogger::outbuf::outbuf(const int& fd,
                         const eo::Levels& contexlvl,
                         const eo::Levels& selectedlvl)
    : _fd(fd), _contextLevel(contexlvl), _selectedLevel(selectedlvl)
{}

int eoLogger::outbuf::overflow(int_type c)
{
    if (_selectedLevel >= _contextLevel)
      {
        if (_fd >= 0 && c != EOF)
          {
              ::write(_fd, &c, 1);
          }
      }
    return c;
}

namespace eo
{
    file::file(const std::string f)
        : _f(f)
    {}

    setlevel::setlevel(const std::string v)
        : _v(v), _lvl((Levels)-1)
    {}

    setlevel::setlevel(const Levels lvl)
        : _v(std::string("")), _lvl(lvl)
    {}
}

void make_verbose(eoParser& parser)
{
    eo::log._createParameters( parser );

    eo::log << eo::setlevel(eo::log._verbose.value());
}

eoLogger eo::log;
