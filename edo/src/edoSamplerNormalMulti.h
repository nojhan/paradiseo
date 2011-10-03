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

#include <edoSampler.h>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/symmetric.hpp>

//! edoSamplerNormalMulti< EOT >

template< class EOT, typename D = edoNormalMulti< EOT > >
class edoSamplerNormalMulti : public edoSampler< D >
{
public:
    typedef typename EOT::AtomType AtomType;

    edoSamplerNormalMulti( edoRepairer<EOT> & repairer ) : edoSampler< D >( repairer) {}

    class Cholesky
    {
    public:
        Cholesky( const ublas::symmetric_matrix< AtomType, ublas::lower >& V)
        {
            unsigned int Vl = V.size1();

            assert(Vl > 0);

            unsigned int Vc = V.size2();

            assert(Vc > 0);

            assert( Vl == Vc );

            _L.resize(Vl);

            unsigned int i,j,k;

            // first column
            i=0;

            // diagonal
            j=0;
            _L(0, 0) = sqrt( V(0, 0) );

            // end of the column
            for ( j = 1; j < Vc; ++j )
            {
                _L(j, 0) = V(0, j) / _L(0, 0);
            }

            // end of the matrix
            for ( i = 1; i < Vl; ++i ) // each column
            {

                // diagonal
                double sum = 0.0;

                for ( k = 0; k < i; ++k)
                {
                    sum += _L(i, k) * _L(i, k);
                }

                _L(i,i) = sqrt( fabs( V(i,i) - sum) );

                for ( j = i + 1; j < Vl; ++j ) // rows
                {
                    // one element
                    sum = 0.0;

                    for ( k = 0; k < i; ++k )
                    {
                        sum += _L(j, k) * _L(i, k);
                    }

                    _L(j, i) = (V(j, i) - sum) / _L(i, i);
                }
            }
        }

        const ublas::symmetric_matrix< AtomType, ublas::lower >& get_L() const {return _L;}

    private:
        ublas::symmetric_matrix< AtomType, ublas::lower > _L;
    };

    edoSamplerNormalMulti( edoBounder< EOT > & bounder )
        : edoSampler< edoNormalMulti< EOT > >( bounder )
    {}

    EOT sample( edoNormalMulti< EOT >& distrib )
    {
        unsigned int size = distrib.size();

        assert(size > 0);


        //-------------------------------------------------------------
        // Cholesky factorisation gererating matrix L from covariance
        // matrix V.
        // We must use cholesky.get_L() to get the resulting matrix.
        //
        // L = cholesky decomposition of varcovar
        //-------------------------------------------------------------

        Cholesky cholesky( distrib.varcovar() );
        ublas::symmetric_matrix< AtomType, ublas::lower > L = cholesky.get_L();

        //-------------------------------------------------------------


        //-------------------------------------------------------------
        // T = vector of size elements drawn in N(0,1) rng.normal(1.0)
        //-------------------------------------------------------------

        ublas::vector< AtomType > T( size );

        for ( unsigned int i = 0; i < size; ++i )
            {
            T( i ) = rng.normal( 1.0 );
            }

        //-------------------------------------------------------------


        //-------------------------------------------------------------
        // LT = prod( L, T )
        //-------------------------------------------------------------

        ublas::vector< AtomType > LT = ublas::prod( L, T );

        //-------------------------------------------------------------


        //-------------------------------------------------------------
        // solution = means + LT
        //-------------------------------------------------------------

        ublas::vector< AtomType > mean = distrib.mean();

        ublas::vector< AtomType > ublas_solution = mean + LT;

        EOT solution( size );

        std::copy( ublas_solution.begin(), ublas_solution.end(), solution.begin() );

        //-------------------------------------------------------------

        return solution;
    }
};

#endif // !_edoSamplerNormalMulti_h
