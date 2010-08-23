//-----------------------------------------------------------------------------
// doFileSnapshot.cpp
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2001
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
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
 */
//-----------------------------------------------------------------------------

#include <cstdlib>

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <doFileSnapshot.h>
#include <utils/compatibility.h>
#include <utils/eoParam.h>

doFileSnapshot::doFileSnapshot(std::string dirname,
			       unsigned int frequency /*= 1*/,
			       std::string filename /*= "gen"*/,
			       std::string delim /*= " "*/,
			       unsigned int counter /*= 0*/,
			       bool rmFiles /*= true*/)
    : _dirname(dirname), _frequency(frequency),
      _filename(filename), _delim(delim),
      _counter(counter), _boolChanged(true)
{
    std::string s = "test -d " + _dirname;

    int res = system(s.c_str());

    // test for (unlikely) errors

    if ( (res == -1) || (res == 127) )
	{
	    throw std::runtime_error("Problem executing test of dir in eoFileSnapshot");
	}

    // now make sure there is a dir without any genXXX file in it
    if (res)                    // no dir present
	{
	    s = std::string("mkdir ") + _dirname;
	}
    else if (!res && rmFiles)
	{
	    s = std::string("/bin/rm ") + _dirname+ "/" + _filename + "*";
	}
    else
	{
	    s = " ";
	}

    int dummy;
    dummy = system(s.c_str());
    // all done
}


void doFileSnapshot::setCurrentFileName()
{
    std::ostringstream oscount;
    oscount << _counter;
    _currentFileName = _dirname + "/" + _filename + oscount.str();
}

eoMonitor& doFileSnapshot::operator()(void)
{
    if (_counter % _frequency)
	{
	    _boolChanged = false;  // subclass with gnuplot will do nothing
	    _counter++;
	    return (*this);
	}
    _counter++;
    _boolChanged = true;
    setCurrentFileName();
    std::ofstream os(_currentFileName.c_str());

    if (!os)
	{
	    std::string str = "doFileSnapshot: Could not open " + _currentFileName;
	    throw std::runtime_error(str);
	}

    return operator()(os);
}

eoMonitor& doFileSnapshot::operator()(std::ostream& os)
{
    iterator it = vec.begin();

    os << (*it)->getValue();

    for ( ++it; it != vec.end(); ++it )
	{
	    os << _delim.c_str() << (*it)->getValue();
	}

    os << '\n';

    return *this;
}
