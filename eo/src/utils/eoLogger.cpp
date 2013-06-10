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

/* TODO?
add a possibility to redirect to several streams
add_redirect/remove_redirect
*/

void eoLogger::_init()
{

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
    std::ostream(NULL),

    _verbose("quiet", "verbose", "Set the verbose level", 'v'),
    _printVerboseLevels(false, "print-verbose-levels", "Print verbose levels", 'l'),
    _output("", "output", "Redirect a standard output to a file", 'o'),

    _selectedLevel(eo::progress),
    _contextLevel(eo::quiet),
    _obuf(_contextLevel, _selectedLevel)
{
    std::ostream::init(&_obuf);
    _init();
}

eoLogger::~eoLogger()
{
	//redirect(NULL);
	if (_obuf._ownedFileStream != NULL) {
		delete _obuf._ownedFileStream;
	}
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
    // we redirect the log to the given filename if -o is used.
    //------------------------------------------------------------------

    if ( ! _output.value().empty() )
        {
    		redirect(_output.value());
        }



    //------------------------------------------------------------------


    //------------------------------------------------------------------
    // we print the list of levels if -l parameter is used.
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

eoLogger& operator<<(eoLogger& l, eo::setlevel v)
{
    l._selectedLevel = (v._lvl < 0 ? l._levels[v._v] : v._lvl);
    return l;
}

eoLogger& operator<<(eoLogger& l, std::ostream& os)
{
    l._obuf._outStream = &os;
    return l;
}

void eoLogger::redirect(std::ostream& os)
{
	if (_obuf._ownedFileStream != NULL) {
		delete _obuf._ownedFileStream;
		_obuf._ownedFileStream = NULL;
	}
    _obuf._outStream = &os;
}

void eoLogger::redirect(const char * filename)
{
	std::ofstream * os;
	if (filename == NULL) {
		os = NULL;
	} else {
		os = new std::ofstream(filename);
	}
	redirect(*os);
	_obuf._ownedFileStream = os;
}

void eoLogger::redirect(const std::string& filename)
{
	redirect(filename.c_str());
}


eoLogger::outbuf::outbuf(const eo::Levels& contexlvl,
                         const eo::Levels& selectedlvl)
    : _outStream(&std::cout), _ownedFileStream(NULL), _contextLevel(contexlvl), _selectedLevel(selectedlvl)
{}

int eoLogger::outbuf::overflow(int_type c)
{
    if (_selectedLevel >= _contextLevel)
      {
        if (_outStream && c != EOF)
          {
              (*_outStream) << (char) c;
          }
      }
    return c;
}

namespace eo
{
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
