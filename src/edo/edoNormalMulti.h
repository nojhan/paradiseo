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

Copyright (C) 2010 Thales group
*/
/*
    Authors:
             Johann Dreo <johann.dreo@thalesgroup.com>
             Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoNormalMulti_h
#define _edoNormalMulti_h

#include "edoDistrib.h"

#ifdef WITH_BOOST
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>
namespace ublas = boost::numeric::ublas;
#else
#ifdef WITH_EIGEN
#include <Eigen/Dense>
#endif // WITH_EIGEN
#endif // WITH_BOOST

/** @defgroup EMNA
 *
 * Estimation of Multivariate Normal Algorithm (EMNA) is a stochastic,
 * derivative-free methods for numerical optimization of non-linear or
 * non-convex continuous optimization problems.
 *
 * @ingroup Algorithms
 */

/** @defgroup Multinormal Multivariate normal
 *
 * Distribution that model co-variances between variables.
 *
 * @ingroup Distributions
 */

/** A multi-normal distribution, that models co-variances.
 *
 * Defines a mean vector and a co-variances matrix.
 *
 * Exists in two implementations, using either
 * <a href="http://www.boost.org/doc/libs/1_50_0/libs/numeric/ublas/doc/index.htm">Boost::uBLAS</a> (if compiled WITH_BOOST)
 * or <a href="http://eigen.tuxfamily.org">Eigen3</a> (WITH_EIGEN).
 *
 * @ingroup Distributions
 * @ingroup EMNA
 * @ingroup Multinormal
 */
template < typename EOT >
class edoNormalMulti : public edoDistrib< EOT >
{
#ifdef WITH_BOOST


public:
    typedef typename EOT::AtomType AtomType;

    edoNormalMulti( unsigned int dim = 1 ) :
            _mean( ublas::vector<AtomType>(0,dim)        ),
        _varcovar( ublas::identity_matrix<AtomType>(dim) )
    {
        assert(_mean.size() > 0);
        assert(_mean.size() == _varcovar.size1());
        assert(_mean.size() == _varcovar.size2());
    }

    edoNormalMulti
    (
     const ublas::vector< AtomType >& mean,
     const ublas::symmetric_matrix< AtomType, ublas::lower >& varcovar
     )
        : _mean(mean), _varcovar(varcovar)
    {
        assert(_mean.size() > 0);
        assert(_mean.size() == _varcovar.size1());
        assert(_mean.size() == _varcovar.size2());
    }

    unsigned int size()
    {
        assert(_mean.size() == _varcovar.size1());
        assert(_mean.size() == _varcovar.size2());
        return _mean.size();
    }

    ublas::vector< AtomType > mean() const {return _mean;}
    ublas::symmetric_matrix< AtomType, ublas::lower > varcovar() const {return _varcovar;}

private:
    ublas::vector< AtomType > _mean;
    ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;

#else
#ifdef WITH_EIGEN


public:
    typedef typename EOT::AtomType AtomType;
    typedef Eigen::Matrix< AtomType, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix< AtomType, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    edoNormalMulti( unsigned int dim = 1 ) :
            _mean( Vector::Zero(dim) ),
        _varcovar( Matrix::Identity(dim,dim) )
    {
        assert(_mean.size() > 0);
        assert(_mean.innerSize() == _varcovar.innerSize());
        assert(_mean.innerSize() == _varcovar.outerSize());
    }

    edoNormalMulti(
        const Vector & mean,
        const Matrix & varcovar
    )
        : _mean(mean), _varcovar(varcovar)
    {
        assert(_mean.innerSize() > 0);
        assert(_mean.innerSize() == _varcovar.innerSize());
        assert(_mean.innerSize() == _varcovar.outerSize());
    }

    unsigned int size()
    {
        assert(_mean.innerSize() == _varcovar.innerSize());
        assert(_mean.innerSize() == _varcovar.outerSize());
        return _mean.innerSize();
    }

    Vector mean() const {return _mean;}
    Matrix varcovar() const {return _varcovar;}

private:
    Vector _mean;
    Matrix _varcovar;

#endif // WITH_EIGEN
#endif // WITH_BOOST

}; // class edoNormalMulti

#endif // !_edoNormalMulti_h
