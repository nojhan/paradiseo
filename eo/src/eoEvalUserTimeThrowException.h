/*
(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

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
*/

#if !defined(_POSIX_VERSION) && !defined(__unix__) && !defined(_WINDOWS)
#warning "Warning: class 'eoEvalUserTimeThrowException' is only available under UNIX (defining 'rusage' in 'sys/resource.h') or Win32 (defining 'GetProcessTimes' in 'WinBase.h') systems, contributions for other systems are welcomed."
#else // defined(_POSIX_VERSION) || defined(__unix__) || defined(_WINDOWS)

#ifndef __EOEVALUSERTIMETHROWEXCEPTION_H__
#define __EOEVALUSERTIMETHROWEXCEPTION_H__

/** Check at each evaluation if a given CPU user time contract has been reached.
 *
 * Throw an eoMaxTimeException if the given max time has been reached.
 * Usefull if you want to end the search independently of generations.
 * This class uses (almost-)POSIX or Win32 headers, depending on the platform.
 * It uses a computation of the user time used on the CPU. For a wallclock time measure, see eoEvalTimeThrowException
 *
 * @ingroup Evaluation
 */

#include "eoExceptions.h"

#if defined(_POSIX_VERSION) || defined(__unix__)

#include <sys/time.h>
#include <sys/resource.h>

template< class EOT >
class eoEvalUserTimeThrowException : public eoEvalFuncCounter< EOT >
{
public:
    eoEvalUserTimeThrowException( eoEvalFunc<EOT> & func, const long max ) : eoEvalFuncCounter<EOT>( func, "CPU-user"), _max(max) {}

    virtual void operator() ( EOT & eo )
    {
        if( eo.invalid() ) {

            getrusage(RUSAGE_SELF,&_usage);

            long current = _usage.ru_utime.tv_sec;
            if( current >= _max ) {
                throw eoMaxTimeException( current );
            } else {
                this->func(eo);
            }
        }
    }

protected:
    const long _max;
    struct rusage _usage;
};

#else
#ifdef _WINDOWS
//here _WINDOWS is defined

#include <WinBase.h>

template< class EOT >
class eoEvalUserTimeThrowException : public eoEvalFuncCounter< EOT >
{
public:
    eoEvalUserTimeThrowException( eoEvalFunc<EOT> & func, const long max ) : eoEvalFuncCounter<EOT>( func, "CPU-user"), _max(max) {}

    virtual void operator() ( EOT & eo )
    {
        if( eo.invalid() ) {
            FILETIME dummy;
            GetProcessTimes(GetCurrentProcess(), &dummy, &dummy, &dummy, &_usage);

            ULARGE_INTEGER current;
            current.LowPart = _usage.dwLowDateTime;
            current.HighPart = _usage.dwHighDateTime;
            if( current.QuadPart >= _max ) {
                throw eoMaxTimeException( current.QuadPart );
            } else {
                func(eo);
            }
        }
    }

protected:
    const long _max;
    FILETIME _usage;
};

#endif // _WINDOWS
#endif // defined(_POSIX_VERSION) || defined(__unix__)
#endif // __EOEVALUSERTIMETHROWEXCEPTION_H__
#endif //!defined(_POSIX_VERSION) && !defined(__unix__) && !defined(_WINDOWS)
