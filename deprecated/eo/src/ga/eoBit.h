/*
   eoBit.h
   (c) GeNeura Team 1998, Marc Schoenauer 2000

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

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@polytechnique.fr
*/

/* MS, Nov. 23, 2000
   Added the calls to base class I/O routines that print the fitness
   Left printing/reading of the size of the bitstring,
       for backward compatibility, and as it is a general practice in EO

   MS, Feb. 7, 2001
   replaced all ...Bin... names with ...Bit... names - for bitstring
   as it was ambiguous with bin...ary things
*/

#ifndef eoBit_h
#define eoBit_h

//-----------------------------------------------------------------------------

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>

#include "eoVector.h"

/** @defgroup bitstring Bit strings

Various functions for a bitstring representation.

Example of a complete test program that use various bitstrings operators:
@include t-eobin.cpp

@ingroup Representations
*/

/** Implementation of bitstring chromosome.

@class eoBit eoBit.h ga/eoBit.h
@ingroup bitstring

Based on STL's std::vector<bool> specialization.
*/
template <class FitT> class eoBit: public eoVector<FitT, bool>
{
public:

    using eoVector< FitT, bool >::begin;
    using eoVector< FitT, bool >::end;
    using eoVector< FitT, bool >::resize;
    using eoVector< FitT, bool >::size;

  /**
   * (Default) Constructor.
   * @param size Size of the binary std::string.
   * @param value Default value.
   */
  eoBit(unsigned size = 0, bool value = false):
    eoVector<FitT, bool>(size, value) {}

  /// My class name.
  virtual std::string className() const
    {
      return "eoBit";
    }

  /**
   * To print me on a stream.
   * @param os The std::ostream.
   */
  virtual void printOn(std::ostream& os) const
    {
      EO<FitT>::printOn(os);
      os << ' ';
      os << size() << ' ';
      std::copy(begin(), end(), std::ostream_iterator<bool>(os));
    }

  /**
   * To read me from a stream.
   * @param is The std::istream.
   */
  virtual void readFrom(std::istream& is)
    {
      EO<FitT>::readFrom(is);
      unsigned s;
      is >> s;
      std::string bits;
      is >> bits;
      if (is)
        {
          resize(bits.size());
          std::transform(bits.begin(), bits.end(), begin(),
                         std::bind2nd(std::equal_to<char>(), '1'));
        }
    }
};

//-----------------------------------------------------------------------------

#endif //eoBit_h


// Local Variables:
// coding: iso-8859-1
// mode: C++
// c-file-offsets: ((c . 0))
// c-file-style: "Stroustrup"
// fill-column: 80
// End:
