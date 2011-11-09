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

#ifndef _edoSamplerNormalMulti_h
#define _edoSamplerNormalMulti_h

#include <cmath>

#include <edoSampler.h>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

/** Sample points in a multi-normal law defined by a mean vector and a covariance matrix.
 *
 * Given M the mean vector and V the covariance matrix, of order n:
 *   - draw a vector T in N(0,I) (i.e. each value is drawn in a normal law with mean=0 an stddev=1)
 *   - compute the Cholesky decomposition L of V (i.e. such as V=LL*)
 *   - return X = M + LT
 */
template< class EOT, typename D = edoNormalMulti< EOT > >
class edoSamplerNormalMulti : public edoSampler< D >
{
public:
    typedef typename EOT::AtomType AtomType;

    edoSamplerNormalMulti( edoRepairer<EOT> & repairer ) : edoSampler< D >( repairer) {}

    /** Cholesky decomposition, given a matrix V, return a matrix L
     * such as V = L Lt (Lt being the conjugate transpose of L).
     *
     * Need a symmetric and positive definite matrix as an input, which 
     * should be the  case of a non-ill-conditionned covariance matrix.
     * Thus, expect a (lower) triangular matrix.
     */
    class Cholesky
    {
    private:
        //! The decomposition is a (lower) symetric matrix, just like the covariance matrix
        ublas::symmetric_matrix< AtomType, ublas::lower > _L;

    public:
        //! The decomposition of the covariance matrix
        const ublas::symmetric_matrix< AtomType, ublas::lower >& decomposition() const {return _L;}

        /** Computation is made at instanciation and then cached in a member variable, 
         * use decomposition() to get the result.
         */
        Cholesky( const ublas::symmetric_matrix< AtomType, ublas::lower >& V )
        {
            factorize( V );
        }

        /** Assert that the covariance matrix have the required properties and returns its dimension.
         *
         * Note: if compiled with NDEBUG, will not assert anything and just return the dimension.
         */
        unsigned assert_properties( const ublas::symmetric_matrix< AtomType, ublas::lower >& V )
        {
            unsigned int Vl = V.size1(); // number of lines
            assert(Vl > 0);

            unsigned int Vc = V.size2(); // number of columns
            assert(Vc > 0);
            assert( Vl == Vc );

            // FIXME assert definite semi-positive

            // the result goes in _L
            _L.resize(Vl);

            return Vl;
        }

        /** This standard algorithm makes use of square root and is thus subject
          * to round-off errors if the covariance matrix is very ill-conditioned.
          */
        void factorize( const ublas::symmetric_matrix< AtomType, ublas::lower >& V)
        {
            unsigned int N = assert_properties( V );

            unsigned int i=0, j=0, k;
            _L(0, 0) = sqrt( V(0, 0) );

            // end of the column
            for ( j = 1; j < N; ++j ) {
                _L(j, 0) = V(0, j) / _L(0, 0);
            }

            // end of the matrix
            for ( i = 1; i < N; ++i ) { // each column
                // diagonal
                double sum = 0.0;
                for ( k = 0; k < i; ++k) {
                    sum += _L(i, k) * _L(i, k);
                }

                // round-off errors may appear here
                assert( V(i,i) - sum >= 0 );
                _L(i,i) = sqrt( V(i,i) - sum );
                //_L(i,i) = sqrt( fabs( V(i,i) - sum) );

                for ( j = i + 1; j < N; ++j ) { // rows
                    // one element
                    sum = 0.0;
                    for ( k = 0; k < i; ++k ) {
                        sum += _L(j, k) * _L(i, k);
                    }

                    _L(j, i) = (V(j, i) - sum) / _L(i, i);

                } // for j in ]i,N[
            } // for i in [1,N[
        }


        /** This alternative algorithm does not use square root BUT the covariance 
         * matrix must be invertible.
         *
         * Computes L and D such as V = L D Lt
         */
        /*
        void factorize_robust( const ublas::symmetric_matrix< AtomType, ublas::lower >& V)
        {
            unsigned int N = assert_properties( V );

            unsigned int i, j, k;
            ublas::symmetric_matrix< AtomType, ublas::lower > D = ublas::zero_matrix<AtomType>(N);
            _L(0, 0) = sqrt( V(0, 0) );

        }
        */


    }; // class Cholesky


    edoSamplerNormalMulti( edoBounder< EOT > & bounder )
        : edoSampler< edoNormalMulti< EOT > >( bounder )
    {}


    EOT sample( edoNormalMulti< EOT >& distrib )
    {
        unsigned int size = distrib.size();
        assert(size > 0);

        // Cholesky factorisation gererating matrix L from covariance
        // matrix V.
        // We must use cholesky.decomposition() to get the resulting matrix.
        //
        // L = cholesky decomposition of varcovar
        Cholesky cholesky( distrib.varcovar() );
        ublas::symmetric_matrix< AtomType, ublas::lower > L = cholesky.decomposition();

        // T = vector of size elements drawn in N(0,1) rng.normal(1.0)
        ublas::vector< AtomType > T( size );
        for ( unsigned int i = 0; i < size; ++i ) {
            T( i ) = rng.normal();
        }

        // LT = L * T
        ublas::vector< AtomType > LT = ublas::prod( L, T );

        // solution = means + LT
        ublas::vector< AtomType > mean = distrib.mean();
        ublas::vector< AtomType > ublas_solution = mean + LT;
        EOT solution( size );
        std::copy( ublas_solution.begin(), ublas_solution.end(), solution.begin() );

        return solution;
    }
};

#endif // !_edoSamplerNormalMulti_h
