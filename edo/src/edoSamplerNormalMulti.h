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


    /** Cholesky decomposition, given a matrix V, return a matrix L
     * such as V = L Lt (Lt being the conjugate transpose of L).
     *
     * Need a symmetric and positive definite matrix as an input, which 
     * should be the case of a non-ill-conditionned covariance matrix.
     * Thus, expect a (lower) triangular matrix.
     */
    class Cholesky
    {
    public:
        typedef ublas::symmetric_matrix< AtomType, ublas::lower > MatrixType;

        enum Method { 
            //! use the standard algorithm, with square root @see factorize_LLT
            standard,
            //! use the algorithm using absolute value within the square root @see factorize_LLT_abs
            absolute, 
            //! use the robust algorithm, without square root @see factorize_LDLT
            robust 
        };
        Method _use;


        /** Instanciate without computing anything, you are responsible of 
         * calling the algorithm and getting the result with operator()
         * */
        Cholesky( Cholesky::Method use = standard ) : _use(use) {}


        /** Computation is made at instanciation and then cached in a member variable, 
         * use decomposition() to get the result.
         *
         * Use the standard unstable method by default.
         */
        Cholesky(const MatrixType& V, Cholesky::Method use = standard ) : _use(use)
        {
            factorize( V );
        }


        /** Compute the factorization and return the result 
         */
        const MatrixType& operator()( const MatrixType& V )
        {
            factorize( V );
            return decomposition();
        }

        //! The decomposition of the covariance matrix
        const MatrixType & decomposition() const {return _L;}

        /** When your using the LDLT robust decomposition (by passing the "robust" 
         * option to the constructor, @see factorize_LDTL), this is the diagonal
         * matrix part.
         */
        const MatrixType & diagonal() const {return _D;}

    protected:

        //! The decomposition is a (lower) symetric matrix, just like the covariance matrix
        MatrixType _L;
       
        //! The diagonal matrix when using the LDLT factorization
        MatrixType _D;


        /** Assert that the covariance matrix have the required properties and returns its dimension.
         *
         * Note: if compiled with NDEBUG, will not assert anything and just return the dimension.
         */
        unsigned assert_properties( const MatrixType& V )
        {
            unsigned int Vl = V.size1(); // number of lines

            // the result goes in _L
            _L.resize(Vl);

#ifndef NDEBUG
            assert(Vl > 0);

            unsigned int Vc = V.size2(); // number of columns
            assert(Vc > 0);
            assert( Vl == Vc );

            // partial assert that V is semi-positive definite
            // assert that all diagonal elements are positives
            for( unsigned int i=0; i < Vl; ++i ) {
                assert( V(i,i) > 0 );
            }

            /* FIXME what is the more efficient way to check semi-positive definite? Candidates are:
                 * perform the cholesky factorization
                 * check if all eigenvalues are positives
                 * check if all of the leading principal minors are positive
                 *
                 */
#endif

            return Vl;
        }


        /** Actually performs the factorization with the method given at 
         * instanciation. Results are cached in _L.
         */
        void factorize( const MatrixType& V )
        {
            if( _use == standard ) {
                factorize_LLT( V );
            } else if( _use == absolute ) {
                factorize_LLT_abs( V );
            } else if( _use == robust ) {
                factorize_LDLT( V );
            }
        }


        /** This standard algorithm makes use of square root and is thus subject
          * to round-off errors if the covariance matrix is very ill-conditioned.
          *
          * When compiled in debug mode and called on ill-conditionned matrix,
          * will raise an assert before calling the square root on a negative number.
          */
        void factorize_LLT( const MatrixType& V)
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


        /** This standard algorithm makes use of square root but do not fail
          * if the covariance matrix is very ill-conditioned.
          * Here, we propagate the error by using the absolute value before
          * computing the square root.
          *
          * Be aware that this increase round-off errors, this is just a ugly
          * hack to avoid crash.
          */
        void factorize_LLT_abs( const MatrixType & V)
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

                _L(i,i) = sqrt( fabs( V(i,i) - sum) );

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


        /** This alternative algorithm do not use square root.
         *
         * Computes L and D such as V = L D Lt
         */
        void factorize_LDLT( const MatrixType& V)
        {
            // use "int" everywhere, because of the "j-1" operation
            int N = assert_properties( V );
            // example of an invertible matrix whose decomposition is undefined
            assert( V(0,0) != 0 ); 

            _D = ublas::zero_matrix<AtomType>(N,N);
            _D(0,0) = V(0,0);
            
            for( int j=0; j<N; ++j ) { // each columns
                _L(j, j) = 1;

                _D(j,j) = V(j,j);
                for( int k=0; k<=j-1; ++k) { // sum
                    _D(j,j) -= _L(j,k) * _L(j,k) * _D(k,k);
                }

                for( int i=j+1; i<N; ++i ) { // remaining rows

                    _L(i,j) = V(i,j);
                    for( int k=0; k<=j-1; ++k) { // sum
                        _L(i,j) -= _L(i,k)*_L(j,k) * _D(k,k);
                    }
                    _L(i,j) /= _D(j,j);

                } // for i in rows
            } // for j in columns
        }
        

    }; // class Cholesky


    edoSamplerNormalMulti( edoRepairer<EOT> & repairer, typename Cholesky::Method use = Cholesky::absolute ) 
        : edoSampler< D >( repairer), _cholesky(use)
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
        const typename Cholesky::MatrixType& L = _cholesky( distrib.varcovar() );

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

protected:
    Cholesky _cholesky;
};

#endif // !_edoSamplerNormalMulti_h
