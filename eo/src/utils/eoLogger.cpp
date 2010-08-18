// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLogger.h
// (c) Thales group, 2010
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

    Authors: 
	     Johann Dréo <johann.dreo@thalesgroup.com>
	     Caner Candan <caner.candan@thalesgroup.com>
 */
//-----------------------------------------------------------------------------


#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstdio> // used to define EOF

#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <locale>

#include "eoLogger.h"

eoLogger	eo::log;

eoLogger::eoLogger()
    : std::ostream(&_obuf),
      _selectedLevel(eo::progress), _contexLevel(eo::quiet),
      _fd(2), _obuf(_fd, _contexLevel, _selectedLevel)
{
    _standard_io_streams[&std::cout] = 1;
    _standard_io_streams[&std::clog] = 2;
    _standard_io_streams[&std::cerr] = 2;

    addLevel("quiet", eo::quiet);
    addLevel("errors", eo::errors);
    addLevel("warnings", eo::warnings);
    addLevel("progress", eo::progress);
    addLevel("logging", eo::logging);
    addLevel("debug", eo::debug);
}

eoLogger::~eoLogger()
{}

std::string	eoLogger::className() const
{
    return ("eoLogger");
}

void	eoLogger::addLevel(std::string name, eo::Levels level)
{
    _levels[name] = level;
    _sortedLevels.push_back(name);
}

void	eoLogger::printLevels() const
{
    std::cout << "Available verbose levels:" << std::endl;

    for (std::vector<std::string>::const_iterator it = _sortedLevels.begin(), end = _sortedLevels.end();
	 it != end; ++it)
	{
	    std::cout << "\t" << *it << std::endl;
	}

    ::exit(0);
}

eoLogger&	operator<<(eoLogger& l, const eo::Levels lvl)
{
    l._contexLevel = lvl;
    return l;
}

eoLogger&	operator<<(eoLogger& l, eo::file f)
{
    l._fd = ::open(f._f.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    return l;
}

eoLogger&	operator<<(eoLogger& l, eo::setlevel v)
{
    l._selectedLevel = (v._lvl < 0 ? l._levels[v._v] : v._lvl);
    return l;
}

eoLogger&	operator<<(eoLogger& l, std::ostream& os)
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
    : _fd(fd), _contexLevel(contexlvl), _selectedLevel(selectedlvl)
{}

int	eoLogger::outbuf::overflow(int_type c)
{
    if (_selectedLevel >= _contexLevel)
      {
	if (_fd >= 0 && c != EOF)
	  {
	      ssize_t	num;
	      num = ::write(_fd, &c, 1);
	  }
      }
    return c;
}

namespace	eo
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
