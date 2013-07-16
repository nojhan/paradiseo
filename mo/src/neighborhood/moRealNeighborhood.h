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

#ifndef __moRealNeighborhood_h__
#define __moRealNeighborhood_h__

#include <mo>
#include "neighborhood/moRealNeighbor.h"

template<class Distrib, class Neighbor>
class moRealNeighborhood : public moRndNeighborhood< Neighbor >
{
public:
    typedef typename Distrib::EOType EOT;

protected:
    Distrib & _distrib;
    edoSampler<Distrib> & _sampler;
    edoBounder<EOT> & _bounder;

public:

    moRealNeighborhood(
        Distrib& distrib,
        edoSampler<Distrib>& sampler,
        edoBounder<EOT>& bounder
    ): _distrib(distrib),
       _sampler(sampler),
       _bounder(bounder)
    { }

    /**
     * It alway remains at least a solution in an infinite neighborhood
     * @param _solution the related solution
     * @return true
     */
    virtual bool hasNeighbor(EOT &)
    {
        return true;
    }

    /**
     * Draw the next neighbor
     * @param _solution the solution to explore
     * @param _current the next neighbor
     */
    virtual void next(EOT &, Neighbor & _current)
    {
        _current.bounder( &_bounder );

        // Draw a translation in the distrib, using the sampler
        _current.translation( _sampler( _distrib ) );
    }


    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _current)
    {
        // there is no difference between an init and a random draw
        next( _solution, _current );
    }
    /**
     * There is always a solution in an infinite neighborhood
     * @param _solution the solution to explore
     * @return true
     */
    virtual bool cont(EOT &)
    {
        return true;
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moRealNeighborhood";
    }

};

#endif // __moRealNeighborhood_h__
