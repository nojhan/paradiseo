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

#ifndef _edoContAdaptiveIllCond_h
#define _edoContAdaptiveIllCond_h

#ifdef WITH_EIGEN

#include<Eigen/Dense>

#include "edoContinue.h"

/** A continuator that check if any matrix among the parameters
 * of an edoNormalAdaptive distribution are ill-conditioned.
 *
 * If the condition number of the covariance matrix
 * or the coordinate system matrix are strictly greater
 * than the threshold given at construction, it will ask for a stop.
 *
 * @ingroup Continuators
 */
template<class D>
class edoContAdaptiveIllCond : public edoContinue<D>
{
public:
    using EOType = typename D::EOType;
    using Matrix = typename D::Matrix;
    using Vector = typename D::Vector;

    edoContAdaptiveIllCond( double threshold = 1e6) :
        _threshold(threshold)
    { }

    bool operator()(const D& d)
    {
        if( condition(d.covar())     > _threshold
         or condition(d.coord_sys()) > _threshold ) {
            return false;
        } else {
            return true;
        }
    }

    virtual std::string className() const { return "edoContAdaptiveIllCond"; }

public:
    // Public function in case someone would want to dimensionate the condition threshold.
    //! Returns the condition number
    bool condition(const Matrix& mat) const
    {
        Eigen::JacobiSVD<Matrix> svd(mat);
        return svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    }

    const double _threshold;
};

#endif // WITH_EIGEN

#endif
