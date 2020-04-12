
/*
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

   © 2020 Thales group

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoAlgoFoundry_H_
#define _eoAlgoFoundry_H_

#include <array>
#include <tuple>

/**
 *
 * @ingroup Core
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT, unsigned NBOP>
class eoAlgoFoundry : public eoAlgo<EOT>
{
    public:
        static const size_t dim = NBOP;

        /** The constructon only take an eval, because all other operators
         * are stored in the public containers.
         */
        eoAlgoFoundry()
        {
            _encoding = { 0 }; // dim * 0
        }

        /** Access to the index of the currently selected operator.
         */
        size_t& at(size_t i)
        {
            return _encoding.at(i);
        }

        /** Select indices of all the operators.
         */
        void operator=( std::array<size_t,dim> a)
        {
            _encoding = a;
        }

    protected:
        std::array<size_t, dim> _encoding;

};

#endif // _eoAlgoFoundry_H_
