/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2001
Copyright (C) 2010 Thales group
*/
/*
Authors:
    todos@geneura.ugr.es
    Marc Schoenauer <Marc.Schoenauer@polytechnique.fr>
    Martin Keijzer <mkeijzer@dhi.dk>
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoPopStat_h
#define _edoPopStat_h

// eo's
#include "../../eo/utils/eoStat.h"


/** Thanks to MS/VC++, eoParam mechanism is unable to handle std::vectors of stats.
This snippet is a workaround:
This class will "print" a whole population into a std::string - that you can later
send to any stream
This is the plain version - see eoPopString for the Sorted version

Note: this Stat should probably be used only within eoStdOutMonitor, and not
inside an eoFileMonitor, as the eoState construct will work much better there.
*/
template <class EOT>
class edoPopStat : public eoStat<EOT, std::string>
{
public:

    using eoStat<EOT, std::string>::value;

    /** default Ctor, void std::string by default, as it appears
	on the description line once at beginning of evolution. and
	is meaningless there. _howMany defaults to 0, that is, the whole
	population*/
    edoPopStat(std::string _desc ="")
	: eoStat<EOT, std::string>("", _desc) {}

    /** Fills the value() of the eoParam with the dump of the population. */
    void operator()(const eoPop<EOT>& _pop)
    {
	std::ostringstream os;
	os << _pop;
	value() = os.str();
    }
};

#endif // !_edoPopStat_h
