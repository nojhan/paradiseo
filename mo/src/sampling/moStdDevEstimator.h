/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Lionel Parreaux <lionel.parreaux@gmail.com>

*/

#ifndef __moStdDevEstimator_h__
#define __moStdDevEstimator_h__

#include <eo>
#include <mo>

#include <continuator/moFitnessVarianceStat.h> // TODO rm
#include <iostream> // TODO rm
 // TODO make tests


template< class EOT, class Neighbor >
class moStdDevEstimator : public eoUF<EOT&, double>
{
public:
	
    /**
     * General constructor for the estimator
     * @param continuator a user-defined continuator
     * @param neighborhood the neighborhood
     * @param fullEval the full evaluation function
     * @param eval neighbor's evaluation function
     * @param walker a local search algorithm
     */
	moStdDevEstimator<EOT,Neighbor> (
			moContinuator<Neighbor>& continuator,
			moNeighborhood < Neighbor > & neighborhood,
			eoEvalFunc<EOT>& fullEval,
			
			/* The following should be read:
			moEval<Neighbor>& eval = _default_eval
			* (which is not possible to achieve as is in C++) */
			const eoOptional< moEval<Neighbor> >& eval = eoOptional< moEval<Neighbor> >::null,
			
			const eoOptional< moLocalSearch<Neighbor> >& walker = eoOptional< moLocalSearch<Neighbor> >::null
		)
	: _default_eval ( fullEval ),
	  _eval(eval.hasValue()? eval.get(): _default_eval),
	  _default_continuator( 0 ),
	  _continuator( _continuator ),
	  _checkpoint( _continuator ),
	  _default_walker( neighborhood, fullEval, _eval, _checkpoint ),
	  _walker( walker.hasValue()? walker.get(): _default_walker )
	{
		_checkpoint.add( _varStat );
	}
	
    /**
     * Simpler constructor for the estimator
     * @param max_iters the number of steps the default moIterContinuator should perform
     * @param neighborhood the neighborhood
     * @param fullEval the full evaluation function
     * @param eval neighbor's evaluation function
     * @param walker a local search algorithm
     */
	moStdDevEstimator<EOT,Neighbor> (
			unsigned int max_iters,
			moNeighborhood < Neighbor > & neighborhood,
			eoEvalFunc<EOT>& fullEval,
			const eoOptional< moEval<Neighbor> >& eval = eoOptional< moEval<Neighbor> >::null,
			const eoOptional< moLocalSearch<Neighbor> >& walker = eoOptional< moLocalSearch<Neighbor> >::null
		)
	: _default_eval ( fullEval ),
	  _eval(eval.hasValue()? eval.get(): _default_eval),
	  _default_continuator( max_iters, false ),
	  _continuator( _default_continuator ),
	  _checkpoint( _continuator ),
	  _default_walker( neighborhood, fullEval, _eval, _checkpoint ),
	  _walker( walker.hasValue()? walker.get(): _default_walker )
	{
		_checkpoint.add( _varStat );
	}

	/**
	 * Evaluates the estimator with the walker algorithm and returns the standard deviation
	 * @param solution the solution from where to start the walk
	 */
	double operator()( EOT & solution )
	{
		_walker(solution);
		return sqrt(_varStat.value());
	}
	
    /**
     * @return the class name
     */
    virtual std::string className(void) const {
        return "moStdDevEstimator";
    }
    
private:
    
    moFullEvalByCopy <Neighbor> _default_eval;
    moEval<Neighbor>& _eval;
	
	moIterContinuator <Neighbor> _default_continuator;
	moContinuator <Neighbor>& _continuator;
	
	moCheckpoint <Neighbor> _checkpoint;
	
    moRandomWalk <Neighbor> _default_walker;
    moLocalSearch <Neighbor> _walker;

	moFitnessVarianceStat<EOT> _varStat;
};

#endif // __moStdDevEstimator_h__


