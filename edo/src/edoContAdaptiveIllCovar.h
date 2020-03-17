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

#ifndef _edoContAdaptiveIllCovar_h
#define _edoContAdaptiveIllCovar_h

#ifdef WITH_EIGEN

#include<Eigen/Dense>

#include "edoContinue.h"

/** A continuator that check if the covariance matrix
 * of an edoNormalAdaptive distribution is ill-conditioned.
 *
 * If the condition number of the covariance matrix
 * is strictly greater than the threshold given at construction,
 * it will ask for a stop.
 *
 * @ingroup Continuators
 */
template<class D>
class edoContAdaptiveIllCovar : public edoContinue<D>
{
public:
    using EOType = typename D::EOType;
    using Matrix = typename D::Matrix;
    using Vector = typename D::Vector;

    edoContAdaptiveIllCovar( double threshold = 1e6) :
        _threshold(threshold)
    { }

    bool operator()(const D& d)
    {
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver( d.covar() );

        auto info = eigensolver.info();
        if(info == Eigen::ComputationInfo::NumericalIssue) {
            eo::log << eo::warnings << "WARNING: the eigen decomposition of the covariance matrix"
                << " did not satisfy the prerequisites." << std::endl;
        } else if(info == Eigen::ComputationInfo::NoConvergence) {
            eo::log << eo::warnings << "WARNING: the eigen decomposition of the covariance matrix"
                << " did not converged." << std::endl;
        } else if(info == Eigen::ComputationInfo::InvalidInput) {
            eo::log << eo::warnings << "WARNING: the eigen decomposition of the covariance matrix"
                << " had invalid inputs." << std::endl;
        }
        if(info != Eigen::ComputationInfo::Success) {
            eo::log << eo::progress << "STOP because the covariance matrix"
               << " cannot be decomposed" << std::endl;
#ifndef NDEBUG
            eo::log << eo::xdebug
                << "mean:\n" << d.mean() << std::endl
                << "sigma:" << d.sigma() << std::endl
                << "coord_sys:\n" << d.coord_sys() << std::endl
                << "scaling:\n" << d.scaling() << std::endl;
#endif
            return false;

        }else {
            Matrix EV = eigensolver.eigenvalues();
            double condition = EV.maxCoeff() / EV.minCoeff();

            if( not std::isfinite(condition) ) {
                eo::log << eo::progress << "STOP because the covariance matrix"
                    << " condition is not finite." << std::endl;
                return false;

            } else if( condition >= _threshold ) {
                eo::log << eo::progress << "STOP because the covariance matrix"
                    << " is ill-conditionned (condition number: " << condition << ")" << std::endl;
                return false;

            } else {
                return true;
            }
        }
    }

    virtual std::string className() const { return "edoContAdaptiveIllCovar"; }

protected:
    const double _threshold;
};

#endif // WITH_EIGEN

#endif
