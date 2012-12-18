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
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/


#ifndef _edoEstimatorNormalMulti_h
#define _edoEstimatorNormalMulti_h


#include "edoEstimator.h"
#include "edoNormalMulti.h"

#ifdef WITH_BOOST
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>
namespace ublas = boost::numeric::ublas;
#else
#ifdef WITH_EIGEN
#include <Eigen/Dense>
#endif // WITH_EIGEN
#endif // WITH_BOOST


/** An estimator for edoNormalMulti
 *
 * Exists in two implementations, using either
 * <a href="http://www.boost.org/doc/libs/1_50_0/libs/numeric/ublas/doc/index.htm">Boost::uBLAS</a> (if compiled WITH_BOOST)
 * or <a href="http://eigen.tuxfamily.org">Eigen3</a> (WITH_EIGEN).
 *
 * @ingroup Estimators
 * @ingroup EMNA
 * @ingroup Multinormal
 */
template < typename EOT, typename D=edoNormalMulti<EOT> >
class edoEstimatorNormalMulti : public edoEstimator<D>
{

#ifdef WITH_BOOST
public:
    class CovMatrix
    {
    public:
        typedef typename EOT::AtomType AtomType;

        CovMatrix( const eoPop< EOT >& pop )
        {
            //-------------------------------------------------------------
            // Some checks before starting to estimate covar
            //-------------------------------------------------------------

            unsigned int p_size = pop.size(); // population size
            assert(p_size > 0);

            unsigned int s_size = pop[0].size(); // solution size
            assert(s_size > 0);

            //-------------------------------------------------------------


            //-------------------------------------------------------------
            // Copy the population to an ublas matrix
            //-------------------------------------------------------------

            ublas::matrix< AtomType > sample( p_size, s_size );

            for (unsigned int i = 0; i < p_size; ++i)
                {
                    for (unsigned int j = 0; j < s_size; ++j)
                        {
                            sample(i, j) = pop[i][j];
                        }
                }

            //-------------------------------------------------------------


            _varcovar.resize(s_size);


            //-------------------------------------------------------------
            // variance-covariance matrix are symmetric (and semi-definite
            // positive), thus a triangular storage is sufficient
            //
            // variance-covariance matrix computation : transpose(A) * A
            //-------------------------------------------------------------

            ublas::symmetric_matrix< AtomType, ublas::lower > var = ublas::prod( ublas::trans( sample ), sample );

            // Be sure that the symmetric matrix got the good size

            assert(var.size1() == s_size);
            assert(var.size2() == s_size);
            assert(var.size1() == _varcovar.size1());
            assert(var.size2() == _varcovar.size2());

            //-------------------------------------------------------------


            // TODO: to remove the comment below

            // for (unsigned int i = 0; i < s_size; ++i)
            //         {
            //             // triangular LOWER matrix, thus j is not going further than i
            //             for (unsigned int j = 0; j <= i; ++j)
            //                 {
            //                     // we want a reducted covariance matrix
            //                     _varcovar(i, j) = var(i, j) / p_size;
            //                 }
            //         }

            _varcovar = var / p_size;

            _mean.resize(s_size); // FIXME: check if it is really used because of the assignation below

            // unit vector
            ublas::scalar_vector< AtomType > u( p_size, 1 );

            // sum over columns
            _mean = ublas::prod( ublas::trans( sample ), u );

            // division by n
            _mean /= p_size;
        }

        const ublas::symmetric_matrix< AtomType, ublas::lower >& get_varcovar() const {return _varcovar;}

        const ublas::vector< AtomType >& get_mean() const {return _mean;}

    private:
        ublas::symmetric_matrix< AtomType, ublas::lower > _varcovar;
        ublas::vector< AtomType > _mean;
    };

public:
    typedef typename EOT::AtomType AtomType;

    edoNormalMulti< EOT > operator()(eoPop<EOT>& pop)
    {
        unsigned int popsize = pop.size();
        assert(popsize > 0);

        unsigned int dimsize = pop[0].size();
        assert(dimsize > 0);

        CovMatrix cov( pop );

        return edoNormalMulti< EOT >( cov.get_mean(), cov.get_varcovar() );
    }
};

#else
#ifdef WITH_EIGEN

public:
    class CovMatrix
    {
    public:
        typedef typename EOT::AtomType AtomType;
        typedef typename D::Vector Vector;
        typedef typename D::Matrix Matrix;

        CovMatrix( const eoPop< EOT >& pop )
        {
            // Some checks before starting to estimate covar
            unsigned int p_size = pop.size(); // population size
            assert(p_size > 0);
            unsigned int s_size = pop[0].size(); // solution size
            assert(s_size > 0);

            // Copy the population to an ublas matrix
            Matrix sample( p_size, s_size );

            for (unsigned int i = 0; i < p_size; ++i) {
                    for (unsigned int j = 0; j < s_size; ++j) {
                            sample(i, j) = pop[i][j];
                    }
            }

            // variance-covariance matrix are symmetric, thus a triangular storage is sufficient
            // variance-covariance matrix computation : transpose(A) * A
            Matrix var = sample.transpose() * sample;

            // Be sure that the symmetric matrix got the good size
            assert(var.innerSize() == s_size);
            assert(var.outerSize() == s_size);

            _varcovar = var / p_size;

            // unit vector
            Vector u( p_size);
            u = Vector::Constant(p_size, 1);

            // sum over columns
            _mean = sample.transpose() * u;

            // division by n
            _mean /= p_size;
        }

        const Matrix& get_varcovar() const {return _varcovar;}

        const Vector& get_mean() const {return _mean;}

    private:
        Matrix _varcovar;
        Vector _mean;
    };

public:
    typedef typename EOT::AtomType AtomType;

    edoNormalMulti< EOT > operator()(eoPop<EOT>& pop)
    {
        unsigned int p_size = pop.size();
        assert(p_size > 0);

        unsigned int s_size = pop[0].size();
        assert(s_size > 0);

        CovMatrix cov( pop );

        assert( cov.get_mean().innerSize() == s_size );
        assert( cov.get_mean().outerSize() == 1 );
        assert( cov.get_varcovar().innerSize() == s_size );
        assert( cov.get_varcovar().outerSize() == s_size );

        return edoNormalMulti< EOT >( cov.get_mean(), cov.get_varcovar() );
    }
#endif // WITH_EIGEN
#endif // WITH_BOOST
}; // class edoNormalMulti

#endif // !_edoEstimatorNormalMulti_h
