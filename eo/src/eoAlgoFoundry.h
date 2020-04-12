
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

#include <vector>

/**
 *
 * @ingroup Core
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT>
class eoAlgoFoundry : public eoAlgo<EOT>
{
    public:
        /** 
         */
        eoAlgoFoundry( size_t nb_operators ) :
            _size(nb_operators),
            _encoding(_size,0)
        { }

        /** Select indices of all the operators.
         */
        void select( std::vector<size_t> encoding )
        {
            assert(encoding.size() == _encoding.size());
            _encoding = encoding;
        }

        /** Access to the index of the currently selected operator.
         */
        size_t& at(size_t i)
        {
            return _encoding.at(i);
        }

        size_t size() const
        {
            return _size;
        }

    protected:
        const size_t _size;
        std::vector<size_t> _encoding;

};

#endif // _eoAlgoFoundry_H_
