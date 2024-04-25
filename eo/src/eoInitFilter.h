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

#ifndef EO_INITFILTER_H
#define EO_INITFILTER_H

// C++ library includes
#include <iostream>
#include <vector>

// EO library includes
#include <eo> // eo general include
#include <eoInit.h>

template<class EOT>
class eoInitFilter : public eoInit<EOT> {

  public:
    typedef bool(*FilterFuncPtr)(const EOT&);

    eoInitFilter(eoInit<EOT>* actualInit_) :
        eoInit<EOT>(), actualInit(actualInit_)
    {
        if (!actualInit_)
            std::cerr << "ERROR: No actual initializer given for eoInitFilter" << std::endl;
    }

    /// My class name
    virtual std::string className() const { return "eoInteractiveInit"; };

    void operator()(EOT& _eo)
    {
        bool accepted = false;

        while (!accepted) {
            (*actualInit)(_eo);

            accepted = true;
            for (FilterFuncPtr fp : filters)
                if ( !(*fp)(_eo) ) {
                    accepted = false;
                    break;
                }
        }
    }

    bool add(FilterFuncPtr fp) {
        if (!fp)
            return false;
        filters.push_back(fp);
        return true;
    }

  private:
    eoInit<EOT>* actualInit;
    std::vector<FilterFuncPtr> filters;

};

#endif // SOPARS_EO_INITFILTER_HPP
