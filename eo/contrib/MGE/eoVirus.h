/* eoVirus.h

(c) GeNeura Team 2001, Marc Schoenauer 2000

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

Contact: todos@geneura.ugr.es, http://geneura.ugr.es
         Marc.Schoenauer@polytechnique.fr
*/


#ifndef eoVirus_h
#define eoVirus_h

#include <iostream>
#include <functional>
#include <string>
#include <vector>

#include "ga/eoBit.h"

/**
\defgroup bitstring

  Various functions for a bitstring representation
*/

/** Implementation of bitstring chromosome.

@class eoBit eoBit.h ga/eoBit.h
@ingroup bitstring

Based on STL's vector<bool> specialization.
*/
template <class FitT>
class eoVirus : public eoBit<FitT>
{
public:

    using eoBit<FitT>::begin;
    using eoBit<FitT>::end;
    using eoBit<FitT>::size;


    /** (Default) Constructor

    @param size Size of the binary std::string.
    */
    eoVirus(unsigned _size = 0, bool _value = false, bool _virValue = false):
        eoBit<FitT>(_size, _value), virus( _size, _virValue) {}

    /// My class name
    virtual std::string className() const {
        return "eoVirus";
    }

    /// Access to virus features
    void virResize( unsigned _i ) {
	virus.resize(_i );
    }

    /// Access to virus features
    bool virusBit( unsigned _i ) const {
	return virus[_i];
    }

    /// Change virus features
    void virusBitSet( unsigned _i, bool _bit ) {
	virus[_i ] = _bit;
    }

    /** To print me on a stream.

    @param os The ostream.
    */
    virtual void printOn(std::ostream& os) const {
        EO<FitT>::printOn(os);
        os << ' ';
        os << size() << ' ';
        std::copy(begin(), end(), std::ostream_iterator<bool>(os));
        std::cout << std::endl;
        std::copy(virus.begin(), virus.end(), std::ostream_iterator<bool>(os));
    }

    /** To read me from a stream.

    @param is The istream.
    */
    virtual void readFrom(std::istream& is){
        eoBit<FitT>::readFrom(is);
        unsigned s;
        is >> s;
        std::string bits;
        is >> bits;
        if (is) {
            virus.resize(bits.size());
            std::transform(bits.begin(), bits.end(), virus.begin(),
#if __cplusplus >= 201103L
                           std::bind(std::equal_to<char>(), std::placeholders::_1, '1'));
#else
                           // Deprecated since C++11.
                           std::bind2nd(std::equal_to<char>(), '1'));
#endif
        }
    }


private:

    std::vector<bool> virus;
};

//-----------------------------------------------------------------------------

#endif //eoBit_h


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-style: "Stroustrup"
// End:
