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

#ifndef EO_FILTERQUADOP_H
#define EO_FILTERQUADOP_H

// C++ library includes
#include <iostream>
#include <vector>

// EO library includes
#include <eo> // eo general include
#include <eoOp.h>

template< class EOT >
class eoFilterQuadOp: public eoQuadOp< EOT >
{
  public:
    typedef bool(*FilterFuncPtr)(const EOT&);

    eoFilterQuadOp(eoQuadOp<EOT>* actualOp_) :
        eoQuadOp< EOT >(), actualOp(actualOp_)
    {}

    virtual ~eoFilterQuadOp() {}

    bool operator()(EOT& _eo1, EOT& _eo2) {
        EOT cpeo1(_eo1);
        EOT cpeo2(_eo2);

        if (!(*actualOp)(cpeo1, cpeo2))
            return false;

        bool accepted = true;
        for (FilterFuncPtr fp : filters)
            if ( !(*fp)(cpeo1) || !(*fp)(cpeo2) ) {
                accepted = false;
                break;
            }

        if (accepted) {
            _eo1 = cpeo1;
            _eo2 = cpeo2;
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
    eoQuadOp<EOT>* actualOp;
    std::vector<FilterFuncPtr> filters;
};

#endif // SOPARS_EO_FILTERQUADOP_HPP
