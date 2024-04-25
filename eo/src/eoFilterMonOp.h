/* Software License Agreement (GNU GPLv3)
 *
 * Copyright (C) 2013  Patrick Lehner <lehner.patrick@gmx.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef EO_FILTERMONOP_H
#define EO_FILTERMONOP_H

// C++ library includes
#include <iostream>
#include <vector>

// EO library includes
#include <eo> // eo general include
#include <eoOp.h>

template< class EOT >
class eoFilterMonOp: public eoMonOp< EOT >
{
  public:
    typedef bool(*FilterFuncPtr)(const EOT&);

    eoFilterMonOp(eoMonOp<EOT>* actualOp_) :
        eoMonOp< EOT >(), actualOp(actualOp_)
    {}

    virtual ~eoFilterMonOp() {}

    bool operator()(EOT& _eo1) {
        EOT eo2(_eo1);

        if (!(*actualOp)(eo2))
            return false;

        bool accepted = true;
        for (FilterFuncPtr fp : filters)
            if ( !(*fp)(eo2) ) {
                accepted = false;
                break;
            }

        if (accepted) {
            _eo1 = eo2;
            return true;
        } else {
            return false;
        }
    }

    bool add(FilterFuncPtr fp) {
        if (!fp)
            return false;
        filters.push_back(fp);
        return true;
    }

  private:
    eoMonOp<EOT>* actualOp;
    std::vector<FilterFuncPtr> filters;
};

#endif // SOPARS_EO_FILTERMONOP_HPP
