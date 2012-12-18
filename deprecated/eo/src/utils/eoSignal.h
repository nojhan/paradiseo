// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/**
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

   Authors: todos@geneura.ugr.es, http://geneura.ugr.es
	    Marc.Schoenauer@polytechnique.fr
	    mak@dhi.dk
	    Caner.Candan@univ-angers.fr
*/

#ifndef _eoSignal_h
#define _eoSignal_h

#include <csignal>
#include <utils/eoCheckPoint.h>
#include <utils/eoLogger.h>

#include <map>
#include <vector>

/**
 * @addtogroup Continuators
 * @{
 */

extern std::map< int, bool > signals_called;

/** eoSignal inherits from eoCheckPoint including signals handling (see signal(7))
 *
 * @ingroup Utilities
 */
template <class EOT>
class eoSignal : public eoCheckPoint<EOT>
{
public :

    eoSignal( int sig = SIGINT ) : eoCheckPoint<EOT>( _dummyContinue ), _sig( sig )
    {
	::signals_called[_sig] = false;

#ifndef _WINDOWS
#ifdef SIGQUIT
	::signal( _sig, handler );
#endif // !SIGQUIT
#endif // !_WINDOWS
    }

    eoSignal( eoContinue<EOT>& _cont, int sig = SIGINT ) : eoCheckPoint<EOT>( _cont ), _sig( sig )
    {
	::signals_called[_sig] = false;

#ifndef _WINDOWS
#ifdef SIGQUIT
	::signal( _sig, handler );
#endif // !SIGQUIT
#endif // !_WINDOWS
    }

    bool operator()( const eoPop<EOT>& _pop )
    {
	bool& called = ::signals_called[_sig];
	if ( called )
	    {
		eo::log << eo::logging << "Signal granted…" << std::endl ;
		called = false;
		return this->eoCheckPoint<EOT>::operator()( _pop );
	    }
	return true;
    }

    virtual std::string className(void) const { return "eoSignal"; }

    static void handler( int sig )
    {
	::signals_called[sig] = true;
	eo::log << eo::logging << "Signal wished…" << std::endl ;
    }

private:
    class DummyContinue : public eoContinue<EOT>
    {
    public:
	bool operator() ( const eoPop<EOT>& ) { return true; }
    } _dummyContinue;

    int _sig;
};

/** @} */

#endif // !_eoSignal_h
