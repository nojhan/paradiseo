// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// doPopStat.h
// (c) Maarten Keijzer, Marc Schoenauer and GeNeura Team, 2001
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

/** WARNING: this file contains 2 classes:

eoPopString and eoSortedPopString

that transform the population into a std::string
that can be used to dump to the screen
*/

#ifndef _doPopStat_h
#define _doPopStat_h

#include <utils/eoStat.h>


/** Thanks to MS/VC++, eoParam mechanism is unable to handle std::vectors of stats.
This snippet is a workaround:
This class will "print" a whole population into a std::string - that you can later
send to any stream
This is the plain version - see eoPopString for the Sorted version

Note: this Stat should probably be used only within eoStdOutMonitor, and not
inside an eoFileMonitor, as the eoState construct will work much better there.
*/
template <class EOT>
class doPopStat : public eoStat<EOT, std::string>
{
public:

    using eoStat<EOT, std::string>::value;

    /** default Ctor, void std::string by default, as it appears
	on the description line once at beginning of evolution. and
	is meaningless there. _howMany defaults to 0, that is, the whole
	population*/
    doPopStat(std::string _desc ="")
	: eoStat<EOT, std::string>("", _desc) {}

    /** Fills the value() of the eoParam with the dump of the population. */
    void operator()(const eoPop<EOT>& _pop)
    {
	std::ostringstream os;
	os << _pop;
	value() = os.str();
    }
};

#endif // !_doPopStat_h
