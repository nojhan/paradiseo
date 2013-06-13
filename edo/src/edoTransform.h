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

#ifndef _edoTransform_h
#define _edoTransform_h

#include <eo> // eoTransform

/** @defgroup Wrappers
 *
 * Wrappers to interact with other parts of the framework
 */

/** Abstract base class for wrapping an estimator and a sampler as an eoTransform
 *
 * @ingroup Wrappers
 */
template<class D>
class edoTransform : public eoTransform< eoPop<typename D::EOType>& >
{
public:
    typedef typename D::EOType EOType;

    edoTransform( edoEstimator<D> & estimator, edoSampler<D> & sampler ) :
        _estimator(estimator), _sampler(sampler)
    {}

    virtual void operator()( eoPop<EOType> & pop ) = 0;

protected:
    edoEstimator<D> & _estimator;
    edoSampler<D> & _sampler;
};


/** Wrapping an estimator and a sampler as an eoTransform.
 *
 * @ingroup Wrappers
 */
template<typename D>
class edoTransformAdaptive : public edoTransform<D>
{
public:
    typedef typename D::EOType EOType;

    edoTransformAdaptive( D & distrib, edoEstimator<D> & estimator, edoSampler<D> & sampler )
        : _distrib(distrib), _estimator(estimator), _sampler(sampler)
    {}

    virtual void operator()( eoPop<EOType> & pop )
    {
        _distrib = _estimator( pop );
        pop.clear();
        for( unsigned int i = 0; i < pop.size(); ++i ) {
            pop.push_back( _sampler(_distrib) );
        }
    }

protected:
    D & _distrib;
    edoEstimator<D> & _estimator;
    edoSampler<D> & _sampler;
};


/** Wrapping an estimator and a sampler as an eoTransform,
 * the distribution is created at instanciation and replaced at each call.
 *
 * @ingroup Wrappers
 */
template<typename D>
class edoTransformStateless : public edoTransformAdaptive<D>
{
public:
    typedef typename D::EOType EOType;

    edoTransformStateless( edoEstimator<D> & estimator, edoSampler<D> & sampler )
        : edoTransformAdaptive<D>( *(new D), estimator, sampler )
    {}

    ~edoTransformStateless()
    {
        // delete the temporary distrib allocated in constructor
        delete &(this->_distrib);
    }
};

#endif // !_edoTransform_h
