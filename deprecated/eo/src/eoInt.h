/*
   eoInt.h
// (c) Marc Schoenauer, Maarten Keijzer and GeNeura Team, 2000

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

    Contact: thomas.legrand@lifl.fr
             todos@geneura.ugr.es, http://geneura.ugr.es
             mkeijzer@dhi.dk
*/

#ifndef eoInt_h
#define eoInt_h

//-----------------------------------------------------------------------------

#include <iostream>    // std::ostream, std::istream
#include <string>      // std::string

#include <eoVector.h>

/** eoInt: implementation of simple integer-valued chromosome.
  * based on eoVector class
  *
  * @ingroup Representations
*/
template <class FitT> class eoInt: public eoVector<FitT, int>
{
 public:

  /**
   * (Default) Constructor.
   * @param size Size of the std::vector
   * @param value fill the vector with this value
   */
  eoInt(unsigned size = 0, int value = 0):
    eoVector<FitT, int>(size, value) {}

  /// My class name.
  virtual std::string className() const
    {
      return "eoInt";
    }

};
/** @example t-eoInt.cpp
 */

//-----------------------------------------------------------------------------

#endif
