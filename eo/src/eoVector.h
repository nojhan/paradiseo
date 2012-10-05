/* (C) GeNeura Team, 2000 - EEAAX 1999 - Maarten Keijzer 2000

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact:     eodev-main@lists.sourceforge.net
Old contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
             mak@dhi.dk
*/

#ifndef _eoVector_h
#define _eoVector_h

#include <vector>
#include <iterator>
#include <EO.h>
#include <utils/eoLogger.h>


/**
 @defgroup Representations Representations

 Solution to a given optimization problem are using a given representation, and are called individuals.

 Some of the most classical representations are proposed, but you can create your own one, providing
 that it inherits from the EO base class. It will be used as a template parameter in almost all operators.

 @{
*/

/** Base class for fixed length chromosomes

It just derives from EO and std::vector and redirects the smaller than
operator to EO (fitness based comparison).

GeneType must have the following methods: void ctor (needed for the
std::vector<>), copy ctor,

*/
template <class FitT, class GeneType>
class eoVector : public EO<FitT>, public std::vector<GeneType>
{
public:

    using EO<FitT>::invalidate;
    using std::vector<GeneType>::operator[];
    using std::vector<GeneType>::begin;
    using std::vector<GeneType>::end;
    using std::vector<GeneType>::resize;
    using std::vector<GeneType>::size;

    typedef GeneType                AtomType;
    typedef std::vector<GeneType>   ContainerType;

    /** default constructor

    @param _size Length of vector (default is 0)
    @param _value Initial value of all elements (default is default value of type GeneType)
    */
    eoVector(unsigned _size = 0, GeneType _value = GeneType())
        : EO<FitT>(), std::vector<GeneType>(_size, _value)
        {}

    /// copy ctor abstracting from the FitT
    template <class OtherFitnessType>
    eoVector(const eoVector<OtherFitnessType, GeneType>& _vec) : std::vector<GeneType>(_vec)
        {}

    // we can't have a Ctor from a std::vector, it would create ambiguity
    //  with the copy Ctor
    void value(const std::vector<GeneType>& _v)
        {
            if (_v.size() != size())       // safety check
            {
                if (size())                // NOT an initial empty std::vector
                    eo::log << eo::warnings << "Warning: Changing size in eoVector assignation" << std::endl;
                resize(_v.size());
            }

            std::copy(_v.begin(), _v.end(), begin());
            invalidate();
        }

    /// to avoid conflicts between EO::operator< and std::vector<GeneType>::operator<
    bool operator<(const eoVector<FitT, GeneType>& _eo) const
        {
            return EO<FitT>::operator<(_eo);
        }

    /// printing...
    virtual void printOn(std::ostream& os) const
        {
            EO<FitT>::printOn(os);
            os << ' ';

            os << size() << ' ';

            std::copy(begin(), end(), std::ostream_iterator<AtomType>(os, " "));
        }

    /// reading...
    virtual void readFrom(std::istream& is)
        {
            EO<FitT>::readFrom(is);

            unsigned sz;
            is >> sz;

            resize(sz);
            unsigned i;

            for (i = 0; i < sz; ++i)
            {
                AtomType atom;
                is >> atom;
                operator[](i) = atom;
            }
        }
};
/** @example t-eoVector.cpp
 */

/** Less than

This is impemented to avoid conflicts between EO::operator< and
std::vector<GeneType>::operator<
*/
template <class FitT, class GeneType>
bool operator<(const eoVector<FitT, GeneType>& _eo1, const eoVector<FitT, GeneType>& _eo2)
{
    return _eo1.operator<(_eo2);
}


/** Greater than

This is impemented to avoid conflicts between EO::operator> and
std::vector<GeneType>::operator>
*/
template <class FitT, class GeneType>
bool operator>(const eoVector<FitT, GeneType>& _eo1, const eoVector<FitT, GeneType>& _eo2)
{
    return _eo1.operator>(_eo2);
}


#endif

/** @} */

// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
