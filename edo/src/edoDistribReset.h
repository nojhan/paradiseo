
/*
The Evolving Objects framework is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own evolutionary algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation;
version 2.1 of the License.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2020 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _EDODISTRIBRESET_H_
#define _EDODISTRIBRESET_H_

#include <eoAlgoReset.h>

/** Reset a distrib when called (as an algorithm).
 *
 * @ingroup Reset
 */
template<class D>
class edoDistribReset : public eoAlgoReset<typename D::EOType>
{
    public:
        using EOType = typename D::EOType;

        edoDistribReset( D& distrib ) :
            _has_dim(false),
            _dim(0),
            _distrib(distrib)
        { }

        edoDistribReset( D& distrib, size_t dim ) :
            _has_dim(true),
            _dim(dim),
            _distrib(distrib)
        { }

        virtual void operator()( eoPop<EOType>& pop )
        {
            if(_has_dim) {
                _distrib.reset(_dim);
            } else {
                _distrib.reset();
            }
        }

    protected:
        bool _has_dim;
        size_t _dim;
        D& _distrib;
};

#endif // _EDODISTRIBRESET_H_
