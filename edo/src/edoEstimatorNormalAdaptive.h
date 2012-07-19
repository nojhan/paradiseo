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
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/


#ifndef _edoEstimatorNormalAdaptive_h
#define _edoEstimatorNormalAdaptive_h

#ifdef WITH_EIGEN

#include <algorithm>

#include<Eigen/Dense>

#include "edoNormalAdaptive.h"
#include "edoEstimatorAdaptive.h"

/** An estimator that works on adaptive normal distributions, basically the heart of the CMA-ES algorithm.
 *
 * @ingroup Estimators
 * @ingroup CMAES
 * @ingroup Adaptivenormal
 */
template< typename EOT, typename D = edoNormalAdaptive<EOT> >
class edoEstimatorNormalAdaptive : public edoEstimatorAdaptive< D >
{
public:
    typedef typename EOT::AtomType AtomType;
    typedef typename D::Vector Vector; // column vectors @see edoNormalAdaptive
    typedef typename D::Matrix Matrix;

    edoEstimatorNormalAdaptive( D& distrib ) :
        edoEstimatorAdaptive<D>( distrib ),
        _calls(0),
        _eigeneval(0)
    {}

private:
    Eigen::VectorXd edoCMAESweights( unsigned int pop_size )
    {
        // compute recombination weights
        Eigen::VectorXd weights( pop_size );
        double sum_w = 0;
        for( unsigned int i = 0; i < pop_size; ++i ) {
            double w_i = log( pop_size + 0.5 ) - log( i + 1 );
            weights(i) = w_i;
            sum_w += w_i;
        }
        // normalization of weights
        weights /= sum_w;

        assert( weights.size() == pop_size);
        return weights;
    }

public:
    void resetCalls()
    {
        _calls = 0;
    }

    // update the distribution reference this->distribution()
    edoNormalAdaptive<EOT> operator()( eoPop<EOT>& pop )
    {

        /**********************************************************************
         * INITIALIZATION
         *********************************************************************/

        unsigned int N = pop[0].size(); // FIXME expliciter la dimension du pb ?
        unsigned int lambda = pop.size();

        // number of calls to the operator == number of generations
        _calls++;
        // number of "evaluations" until now
        unsigned int counteval = _calls * lambda;

        // Here, if we are in canonical CMA-ES,
        // pop is supposed to be the mu ranked better solutions,
        // as the rank mu selection is supposed to have occured.
        Matrix arx( N, lambda );

        // copy the pop (most probably a vector of vectors) in a Eigen3 matrix
        for( unsigned int d = 0; d < N; ++d ) {
            for( unsigned int i = 0; i < lambda; ++i ) {
                arx(d,i) = pop[i][d]; // NOTE: pop = arx.transpose()
            } // dimensions
        } // individuals

        // muXone array for weighted recombination
        Eigen::VectorXd weights = edoCMAESweights( lambda );
        assert( weights.size() == lambda );

        // FIXME exposer les constantes dans l'interface

        // variance-effectiveness of sum w_i x_i
        double mueff = pow(weights.sum(), 2) / (weights.array().square()).sum();

        // time constant for cumulation for C
        double cc = (4+mueff/N) / (N+4 + 2*mueff/N);

        // t-const for cumulation for sigma control
        double cs = (mueff+2) / (N+mueff+5);

        // learning rate for rank-one update of C
        double c1 = 2 / (pow(N+1.3,2)+mueff);

        // and for rank-mu update
        double cmu = 2 * (mueff-2+1/mueff) / ( pow(N+2,2)+mueff);

        // damping for sigma
        double damps = 1 + 2*std::max(0.0, sqrt((mueff-1)/(N+1))-1) + cs;


        // shortcut to the referenced distribution
        D& d = this->distribution();

        // C^-1/2
        Matrix invsqrtC =
            d.coord_sys() * d.scaling().asDiagonal().inverse()
            * d.coord_sys().transpose();
        assert( invsqrtC.innerSize() == d.coord_sys().innerSize() );
        assert( invsqrtC.outerSize() == d.coord_sys().outerSize() );

        // expectation of ||N(0,I)|| == norm(randn(N,1))
        double chiN = sqrt(N)*(1-1/(4*N)+1/(21*pow(N,2)));


        /**********************************************************************
         * WEIGHTED MEAN
         *********************************************************************/

        // compute weighted mean into xmean
        Vector xold = d.mean();
        assert( xold.size() == N );
        //  xmean ( N, 1 ) = arx( N, lambda ) * weights( lambda, 1 )
        Vector xmean = arx * weights;
        assert( xmean.size() == N );
        d.mean( xmean );


        /**********************************************************************
         * CUMULATION: UPDATE EVOLUTION PATHS
         *********************************************************************/

        // cumulation for sigma
        d.path_sigma(
            (1.0-cs)*d.path_sigma() + sqrt(cs*(2.0-cs)*mueff)*invsqrtC*(xmean-xold)/d.sigma()
        );

        // sign of h
        double hsig;
        if( d.path_sigma().norm()/sqrt(1.0-pow((1.0-cs),(2.0*counteval/lambda)))/chiN
                < 1.4 + 2.0/(N+1.0)
          ) {
            hsig = 1.0;
        } else {
            hsig = 0.0;
        }

        // cumulation for the covariance matrix
        d.path_covar(
            (1.0-cc)*d.path_covar() + hsig*sqrt(cc*(2.0-cc)*mueff)*(xmean-xold) / d.sigma()
        );

        Matrix xmu( N, lambda);
        xmu = xold.rowwise().replicate(lambda);
        assert( xmu.innerSize() == N );
        assert( xmu.outerSize() == lambda );
        Matrix artmp = (1.0/d.sigma()) * (arx - xmu);
        // Matrix artmp = (1.0/d.sigma()) * arx - xold.colwise().replicate(lambda);
        assert( artmp.innerSize() == N && artmp.outerSize() == lambda );


        /**********************************************************************
         * COVARIANCE MATRIX ADAPTATION
         *********************************************************************/

        d.covar(
                (1-c1-cmu) * d.covar()                            // regard old matrix
                + c1 * (d.path_covar()*d.path_covar().transpose() // plus rank one update
                    + (1-hsig) * cc*(2-cc) * d.covar())   // minor correction if hsig==0
                + cmu * artmp * weights.asDiagonal() * artmp.transpose() // plus rank mu update
               );

        // Adapt step size sigma
        d.sigma( d.sigma() * exp((cs/damps)*(d.path_sigma().norm()/chiN - 1)) );



        /**********************************************************************
         * DECOMPOSITION OF THE COVARIANCE MATRIX
         *********************************************************************/

        // Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        if( counteval - _eigeneval > lambda/(c1+cmu)/N/10 ) {  // to achieve O(N^2)
            _eigeneval = counteval;

            // enforce symmetry of the covariance matrix
            Matrix C = d.covar();
            // FIXME edoEstimatorNormalAdaptive.h:213:44: erreur: expected primary-expression before ‘)’ token
            // copy the upper part in the lower one
            //C.triangularView<Eigen::Lower>() = C.adjoint();
            // Matrix CS = C.triangularView<Eigen::Upper>() + C.triangularView<Eigen::StrictlyUpper>().transpose();
            d.covar( C );

            Eigen::SelfAdjointEigenSolver<Matrix> eigensolver( d.covar() ); // FIXME use JacobiSVD?
            d.coord_sys( eigensolver.eigenvectors() );
            Matrix mD = eigensolver.eigenvalues().asDiagonal();
            assert( mD.innerSize() == N && mD.outerSize() == N );

            // from variance to standard deviations
            mD.cwiseSqrt();
            d.scaling( mD.diagonal() );
        }

        return d;
    } // operator()

protected:

    unsigned int _calls;
    unsigned int _eigeneval;


    // D & distribution() inherited from edoEstimatorAdaptive
};
#endif // WITH_EIGEN

#endif // !_edoEstimatorNormalAdaptive_h
