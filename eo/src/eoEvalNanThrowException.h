/*
(c) Thales group, 2020

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

#ifndef __eoEvalNanThrowException_h__
#define __eoEvalNanThrowException_h__

#include <cmath>

#include "eoEvalFunc.h"
#include "eoExceptions.h"

/*!
Wrap an evaluation function so that an exception may be thrown when the
eval function returns a bad value (Not A Number or infinity).

@warning: Only work for eoScalarFitness.

The class throw an eoNanException. You can catch this exception
from your main function, so as to stop everything properly.

@ingroup Evaluation
*/
template < typename EOT >
class eoEvalNanThrowException : public eoEvalFunc< EOT >
{
public :
    eoEvalNanThrowException( eoEvalFunc<EOT>& func)
        : _func(func)
    {}

    //! Evaluate the individual, then throw an exception if it exceed the max number of evals.
    virtual void operator()(EOT& sol)
    {
        if(sol.invalid()) {
            _func(sol);

            if(not std::isfinite(sol.fitness()) ) {
#ifndef NDEBUG
                eo::log << eo::xdebug << sol << std::endl;
#endif
                throw eoNanException();
            }
        } // if invalid
    }

    virtual std::string className() const {return "eoEvalNanThrowException";}

protected:
    eoEvalFunc<EOT>& _func;
};

#endif // __eoEvalNanThrowException_h__
