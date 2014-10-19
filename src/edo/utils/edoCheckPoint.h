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

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoCheckPoint_h
#define _edoCheckPoint_h

// eo's
#include "../../eo/utils/eoUpdater.h"
#include "../../eo/utils/eoMonitor.h"

#include "../edoContinue.h"
#include "edoStat.h"

//! eoCheckPoint< EOT > classe fitted to Distribution Object library

template < typename D >
class edoCheckPoint : public edoContinue< D >
{
public:
    typedef typename D::EOType EOType;

    edoCheckPoint(edoContinue< D >& _cont)
    {
	_continuators.push_back( &_cont );
    }

    bool operator()(const D& distrib)
    {
	for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
	    {
		(*_stats[i])( distrib );
	    }

	for ( unsigned int i = 0, size = _updaters.size(); i < size; ++i )
	    {
		(*_updaters[i])();
	    }

	for ( unsigned int i = 0, size = _monitors.size(); i < size; ++i )
	    {
		(*_monitors[i])();
	    }

	bool bContinue = true;
	for ( unsigned int i = 0, size = _continuators.size(); i < size; ++i )
	    {
		if ( !(*_continuators[i])( distrib ) )
		    {
			bContinue = false;
		    }
	    }

	if ( !bContinue )
	    {
		for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
		    {
			_stats[i]->lastCall( distrib );
		    }

		for ( unsigned int i = 0, size = _updaters.size(); i < size; ++i )
		    {
			_updaters[i]->lastCall();
		    }

		for ( unsigned int i = 0, size = _monitors.size(); i < size; ++i )
		    {
			_monitors[i]->lastCall();
		    }
	    }

	return bContinue;
    }

    void add(edoContinue< D >& cont) { _continuators.push_back( &cont ); }
    void add(edoStatBase< D >& stat) { _stats.push_back( &stat ); }
    void add(eoMonitor& mon) { _monitors.push_back( &mon ); }
    void add(eoUpdater& upd) { _updaters.push_back( &upd ); }

    virtual std::string className(void) const { return "edoCheckPoint"; }

    std::string allClassNames() const
    {
	std::string s("\n" + className() + "\n");

	s += "Stats\n";
	for ( unsigned int i = 0, size = _stats.size(); i < size; ++i )
	    {
		s += _stats[i]->className() + "\n";
	    }
	s += "\n";

	s += "Updaters\n";
	for ( unsigned int i = 0; i < _updaters.size(); ++i )
	    {
		s += _updaters[i]->className() + "\n";
	    }
	s += "\n";

	s += "Monitors\n";
	for ( unsigned int i = 0; i < _monitors.size(); ++i )
	    {
		s += _monitors[i]->className() + "\n";
	    }
	s += "\n";

	s += "Continuators\n";
	for ( unsigned int i = 0, size = _continuators.size(); i < size; ++i )
	    {
		s += _continuators[i]->className() + "\n";
	    }
	s += "\n";

	return s;
    }

private:
    std::vector< edoContinue< D >* > _continuators;
    std::vector< edoStatBase< D >* > _stats;
    std::vector< eoMonitor* > _monitors;
    std::vector< eoUpdater* > _updaters;
};

#endif // !_edoCheckPoint_h
