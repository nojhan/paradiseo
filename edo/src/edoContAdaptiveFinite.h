/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

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

#ifndef _edoContAdaptiveFinite_h
#define _edoContAdaptiveFinite_h

#include "edoContinue.h"

/** A continuator that check if any element in the parameters
 * of an edoNormalAdaptive distribution are finite
 *
 * If any element of any parameter is infinity or NaN (Not A Number),
 * it will ask for a stop.
 *
 * @ingroup Continuators
 */
template<class D>
class edoContAdaptiveFinite : public edoContinue<D>
{
public:
    using EOType = typename D::EOType;
    using Matrix = typename D::Matrix;
    using Vector = typename D::Vector;

    bool operator()(const D& d)
    {
        // Try to finite_check in most probably ill-conditioned order.
        return finite_check(d.covar())
           and finite_check(d.path_covar())
           and finite_check(d.coord_sys())
           and finite_check(d.scaling())
           and finite_check(d.path_sigma())
           and finite_check(d.sigma())
           ;
    }

    virtual std::string className() const { return "edoContAdaptiveFinite"; }

protected:
    bool finite_check(const Matrix& mat) const
    {
        for(long i=0; i<mat.rows(); ++i) {
            for(long j=0; j<mat.cols(); ++j) {
                if(not finite_check(mat(i,j))) {
                    return false;
                }
            }
        }
        return true;
    }

    bool finite_check(const Vector& vec) const
    {
        for(long i=0; i<vec.size(); ++i) {
            if(not finite_check(vec[i])) {
                return false;
            }
        }
        return true;
    }

    bool finite_check(const typename EOType::AtomType& x) const
    {
        if(not std::isfinite(x)) {
            return false;
        } else {
            return true;
        }
    }
};

#endif
