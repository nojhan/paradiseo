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
        bool fin_sigma      = is_finite(d.sigma()     );
        bool fin_path_sigma = is_finite(d.path_sigma());
        bool fin_scaling    = is_finite(d.scaling()   );
        bool fin_coord_sys  = is_finite(d.coord_sys() );
        bool fin_path_covar = is_finite(d.path_covar());
        bool fin_covar      = is_finite(d.covar()     );

        bool all_finite = fin_covar
           and fin_path_covar
           and fin_coord_sys
           and fin_scaling
           and fin_path_sigma
           and fin_sigma;

        if( not all_finite ) {
            eo::log << eo::progress << "STOP because parameters are not finite: ";
            if( not fin_covar      ) { eo::log << eo::errors << "covar, "; }
            if( not fin_path_covar ) { eo::log << eo::errors << "path_covar, "; }
            if( not fin_coord_sys  ) { eo::log << eo::errors << "coord_sys, "; }
            if( not fin_scaling    ) { eo::log << eo::errors << "scaling, "; }
            if( not fin_path_sigma ) { eo::log << eo::errors << "path_sigma, "; }
            if( not fin_sigma      ) { eo::log << eo::errors << "sigma"; }
            eo::log << eo::errors << std::endl;
        }
        return all_finite;
    }

    virtual std::string className() const { return "edoContAdaptiveFinite"; }

protected:
    bool is_finite(const Matrix& mat) const
    {
        for(long i=0; i<mat.rows(); ++i) {
            for(long j=0; j<mat.cols(); ++j) {
                // Double negation because one want to escape
                // as soon as one element is not finite.
                if(not is_finite(mat(i,j))) {
                    return false;
                }
            }
        }
        return true;
    }

    bool is_finite(const Vector& vec) const
    {
        for(long i=0; i<vec.size(); ++i) {
            if(not is_finite(vec[i])) {
                return false;
            }
        }
        return true;
    }

    bool is_finite(const typename EOType::AtomType& x) const
    {
        if(std::isfinite(x)) {
            return true;
        } else {
            return false;
        }
    }
};

#endif
