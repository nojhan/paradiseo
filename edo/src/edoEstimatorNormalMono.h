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

#ifndef _edoEstimatorNormalMono_h
#define _edoEstimatorNormalMono_h

#include "edoEstimator.h"
#include "edoNormalMono.h"

/** An estimator for edoNormalMono
 *
 * @ingroup Estimators
 * @ingroup Mononormal
 */
template < typename EOT >
class edoEstimatorNormalMono : public edoEstimator< edoNormalMono< EOT > >
{
    public:
        typedef typename EOT::AtomType AtomType;

        //! Knuth's algorithm, online variance, numericably stable
        class Variance
        {
            public:
                Variance() : _n(0), _mean(0), _M2(0) {}

                void update(AtomType x)
                {
                    _n++;

                    AtomType delta = x - _mean;

                    _mean += delta / _n;
                    _M2 += delta * ( x - _mean );
                }

                AtomType mean() const {return _mean;}

                //! Population variance
                AtomType var_n() const {
                    assert( _n > 0 );
                    return _M2 / _n;
                }

                /** Sample variance (using Bessel's correction)
                 * is an unbiased estimate of the population variance,
                 * but it has uniformly higher mean squared error
                 */
                AtomType var() const {
                    assert( _n > 1 );
                    return _M2 / (_n - 1);
                }

                //! Population standard deviation
                AtomType std_n() const {return sqrt( var_n() );}

                //! Sample standard deviation, is a biased estimate of the population standard deviation
                AtomType std() const {return sqrt( var() );}

            private:
                AtomType _n;
                AtomType _mean;
                AtomType _M2;
        };

    public:
        edoNormalMono< EOT > operator()(eoPop<EOT>& pop)
        {
            unsigned int popsize = pop.size();
            assert(popsize > 0);

            unsigned int dimsize = pop[0].size();
            assert(dimsize > 0);

            std::vector< Variance > var( dimsize );

            for (unsigned int i = 0; i < popsize; ++i)
            {
                for (unsigned int d = 0; d < dimsize; ++d)
                {
                    var[d].update( pop[i][d] );
                }
            }

            EOT mean( dimsize );
            EOT variance( dimsize );

            for (unsigned int d = 0; d < dimsize; ++d)
            {
                mean[d] = var[d].mean();
                variance[d] = var[d].var_n();
            }

            return edoNormalMono< EOT >( mean, variance );
        }
};

#endif // !_edoEstimatorNormalMono_h
