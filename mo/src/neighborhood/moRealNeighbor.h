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

#ifndef __moRealNeighbor_h__
#define __moRealNeighbor_h__

#include <mo>
#include <eo>
#include <edo>

//! A neighbor as produced by a moRealNeighborhood
/*!
 * In a real neighborhood, the move is just a translation vector, of the same type than a solution.
 */

template <class EOT, class Fitness=typename EOT::Fitness>
class moRealNeighbor : public moNeighbor<EOT, Fitness>
{
protected:
    //! The move to be applied
    EOT _translation;

    edoBounder<EOT> * _bounder;


public:

    moRealNeighbor<EOT,Fitness>() : _bounder( NULL ) {  }

    //! Returns the solution attached to this neighbor
    EOT translation() { return _translation; }

    //! Set the translation
    void translation( EOT translation ) { _translation = translation; }


    void bounder( edoBounder<EOT> * bounder ) { _bounder = bounder; }

    /**
     * Assignment operator
     * @param _neighbor the neighbor to assign
     * @return a neighbor equal to the other
     */
    virtual moNeighbor<EOT, Fitness>& operator=(const moNeighbor<EOT, Fitness>& _neighbor) {
        fitness( _neighbor.fitness() );
        return (*this);
    }

    /*!
     * Move a solution to the solution of this neighbor
     * @param _solution the related solution
     */
    virtual void move(EOT & _solution)
    {
        assert( _solution.size() == _translation.size() );

        for( unsigned int i=0, size= _solution.size(); i<size; ++i ) {
            _solution[i] += _translation[i];
        }


        if( _bounder != NULL ) {
            (*_bounder)( _solution );
        }

	_solution.invalidate();
    }


    /**
     * Test equality between two neighbors
     * @param _neighbor a neighbor
     * @return if _neighbor and this one are equals
     */
    virtual bool equals(moRealNeighbor<EOT>& _neighbor) {
        return _neighbor.translation() == _translation;
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moRealNeighbor";
    }
};


#endif // __moRealNeighbor_h__

