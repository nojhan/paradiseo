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

Copyright (C) 2013 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _edoEstimatorBinomialMulti_h
#define _edoEstimatorBinomialMulti_h

#include "edoBinomialMulti.h"
#include "edoEstimator.h"

#ifdef WITH_EIGEN
#include <Eigen/Dense>

/** An estimator for edoBinomialMulti
 *
 * @ingroup Estimators
 * @ingroup Binomial
 */
template< class EOT, class D = edoBinomialMulti<EOT> >
class edoEstimatorBinomialMulti : public edoEstimator<D>
{
protected:
    /** Decide whether a given element of the distribution is true or false.
     *
     * The default implementation is to set the item to the value of the atom itself
     * (which is a boolean in the basic version).
     * If you have a more complex data structure, you can just overload this.
     */
    virtual void make( D & to, unsigned int i, unsigned int j, typename EOT::AtomType::const_iterator iatom )
    {
        to(i,j) = *iatom;
    }

    /** Transliterate a EOT in a boolean matrix
     */
    D eot2d( const EOT & from, unsigned int rows, unsigned int cols ) // FIXME maybe more elegant with Eigen::Map?
    {
        assert( rows > 0 );
        assert( from.size() == rows );
        assert( cols > 0 );

        D to( Eigen::MatrixXd(rows, cols) );
        unsigned int i=0;
        for( typename EOT::const_iterator irow = from.begin(), end=from.end(); irow != end; ++irow ) {
            assert( irow->size() == cols );
            unsigned int j=0;
            for( typename EOT::AtomType::const_iterator icol = irow->begin(), end=irow->end(); icol != end; ++icol ) {
                make( to, i, j, icol );
                j++;
            }
            i++;
        }

        return to;
    }

public:
    /** The expected EOT interface is the same as an Eigen3::MatrixXd.
     */
    D operator()( eoPop<EOT>& pop )
    {
        unsigned int popsize = pop.size();
        assert(popsize > 0);

        unsigned int rows = pop.begin()->size();
        assert( rows > 0 );
        unsigned int cols = pop.begin()->begin()->size();
        assert( cols > 0 );

        D probas( D::Zero(rows, cols) );

        // We still need a loop over pop, because it is an eoVector
        for (unsigned int i = 0; i < popsize; ++i) {
            D indiv = eot2d( pop[i], rows, cols );
            assert( indiv.rows() == rows && indiv.cols() == cols );

            // the EOT matrix should be filled with 1 or 0 only
            assert( indiv.sum() <= popsize );

            probas += indiv / popsize;

            // sum and scalar product, no size pb expected
            assert( probas.rows() == rows && probas.cols() == cols );
        }

        return probas;
    }
};

#endif // WITH_EIGEN
#endif // !_edoEstimatorBinomialMulti_h

