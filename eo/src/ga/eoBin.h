/*
   eoBin.h
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
*/

#ifndef eoBin_h
#define eoBin_h

//-----------------------------------------------------------------------------

#include <iostream>    // ostream, istream
#include <functional>  // bind2nd
#include <string>      // string

#include <eoFixedLength.h>

/**
\defgroup bitstring

  Various functions for a bitstring representation
*/

/** eoBin: implementation of binary chromosome.                               
\class eoBin eoBin.h ga/eoBin.h
\ingroup bitstring
  * based on STL's vector<bool> specialization.                                 
*/
template <class F> class eoBin: public eoFixedLength<F, bool>
{
 public:

  /**
   * (Default) Constructor.
   * @param size Size of the binary string.
   */
  eoBin(unsigned size = 0, bool value = false): 
    eoFixedLength<F, bool>(size, value) {}
      
  /// My class name.
  string className() const 
    { 
      return "eoBin"; 
    }
  
  /**
   * To print me on a stream.
   * @param os The ostream.
   */
  void printOn(ostream& os) const
    {
      EO<F>::printOn(os);
      os << ' ';
      os << size() << ' '; 
      copy(begin(), end(), ostream_iterator<bool>(os));
    }
  
  /**
   * To read me from a stream.
   * @param is The istream.
   */
  void readFrom(istream& is)
    {
      EO<F>::readFrom(is);
      unsigned s;
      is >> s;
      string bits;
      is >> bits;
      if (is)
	{
	  resize(bits.size());
	  transform(bits.begin(), bits.end(), begin(), 
		    bind2nd(equal_to<char>(), '1'));
	}
    }
};

//-----------------------------------------------------------------------------

#endif //eoBin_h
