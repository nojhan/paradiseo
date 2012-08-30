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

#include <ctime>

#include <eoExceptions.h>

/** Check at each evaluation if a given tie contract has been reached.
 *
 * Throw an eoMaxTimeException if the given max time has been reached.
 * Usefull if you want to end the search independently of generations.
 *
 * @ingroup Evaluation
 */
template< class EOT >
class eoEvalTimeThrowException : public eoEvalFuncCounter< EOT >
{
public:
    eoEvalTimeThrowException( eoEvalFunc<EOT> & func, time_t max ) : _max(max), _start( std::time(NULL) ), eoEvalFuncCounter<EOT>( func, "Eval.") {}

    virtual void operator() ( EOT & eo )
    {
        if( eo.invalid() ) {

            time_t elapsed = static_cast<time_t>( std::difftime( std::time(NULL) , _start ) );

            if( elapsed >= _max ) {
                throw eoMaxTimeException(elapsed);
            } else {
                func(eo);
            }
        }
    }

protected:
    time_t _max;

    time_t _start;
};
