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

#include <sys/time.h>
#include <sys/resource.h>

#include <eoExceptions.h>

/** Check at each evaluation if a given CPU user time contract has been reached.
 *
 * Throw an eoMaxTimeException if the given max time has been reached.
 * Usefull if you want to end the search independently of generations.
 * This class uses (almost-)POSIX headers.
 * It uses a computation of the user time used on the CPU. For a wallclock time measure, see eoEvalTimeThrowException
 *
 * @ingroup Evaluation
 */
template< class EOT >
class eoEvalUserTimeThrowException : public eoEvalFuncCounter< EOT >
{
public:
    eoEvalUserTimeThrowException( eoEvalFunc<EOT> & func, long max ) : _max(max), eoEvalFuncCounter<EOT>( func, "CPU-user") {}

    virtual void operator() ( EOT & eo )
    {
        if( eo.invalid() ) {

            getrusage(RUSAGE_SELF,&_usage);

            if( _usage.ru_utime.tv_sec >= _max ) {
                throw eoMaxTimeException( _usage.ru_utime.tv_sec );
            } else {
                func(eo);
            }
        }
    }

protected:
    long _max;
    struct rusage _usage;
};
