/*
   eoVirus.h
   (c) GeNeura Team 2001, Marc Schoenauer 2000

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
CVS Info: $Date: 2001-05-10 12:16:00 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/MGE/Attic/eoVirus.h,v 1.1 2001-05-10 12:16:00 jmerelo Exp $ $Author: jmerelo $
*/

#ifndef eoVirus_h
#define eoVirus_h

//-----------------------------------------------------------------------------

#include <iostream>    // ostream, istream
#include <functional>  // bind2nd
#include <string>      // string

#include <ga/eoBit.h>

/**
\defgroup bitstring

  Various functions for a bitstring representation
*/

/** eoBit: implementation of bitstring chromosome.
\class eoBit eoBit.h ga/eoBit.h
\ingroup bitstring
  * based on STL's vector<bool> specialization.
*/
template <class FitT> class eoVirus: public eoBit<FitT>
{
 public:

  /**
   * (Default) Constructor.
   * @param size Size of the binary string.
   */
  eoVirus(unsigned _size = 0, bool _value = false, bool _virValue = false):
    eoBit<FitT>(_size, _value), virus( _size, _virValue) {}

  /// My class name.
  virtual string className() const {
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

  /**
   * To print me on a stream.
   * @param os The ostream.
   */
  virtual void printOn(ostream& os) const {
      EO<FitT>::printOn(os);
      os << ' ';
      os << size() << ' ';
      copy(begin(), end(), ostream_iterator<bool>(os));
	  cout << endl;
	  copy(virus.begin(), virus.end(), ostream_iterator<bool>(os));
  }

  /**
   * To read me from a stream.
   * @param is The istream.
   */
  virtual void readFrom(istream& is){
      eoBit<FitT>::readFrom(is);
      unsigned s;
      is >> s;
      string bits;
      is >> bits;
      if (is) {
		virus.resize(bits.size());
		transform(bits.begin(), bits.end(), virus.begin(),
				  bind2nd(equal_to<char>(), '1'));
	  }
    }
 private:
  vector<bool> virus;
};

//-----------------------------------------------------------------------------

#endif //eoBit_h
