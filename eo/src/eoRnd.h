/* -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

-----------------------------------------------------------------------------
 eoRnd.h
 (c) GeNeura Team, 1998
 
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
 */
//-----------------------------------------------------------------------------

#ifndef _EORND_H
#define _EORND_H

//-----------------------------------------------------------------------------

#include <eoObject.h>

//-----------------------------------------------------------------------------
// Class eoRnd
//-----------------------------------------------------------------------------

#include <stdlib.h>   // srand
#include <time.h>     // time
#include <stdexcept>  // runtime_error

//-----------------------------------------------------------------------------

#include <eoPersistent.h>

//-----------------------------------------------------------------------------
/** 
 * Base class for a family of random number generators. Generates numbers 
 * according to the parameters given in the constructor. Uses the machine's 
 * own random number generators. Which is not too good, after all. 
 */
template<class T>
class eoRnd: public eoObject, public eoPersistent
{
public:

  /// default constructor
  eoRnd(unsigned _seed = 0) { 
    if ( !started ) {
      srand(_seed?_seed:time(0));
      started = true;
    }
  }
    
  /// Copy cotor
  eoRnd(const eoRnd& _r ) {srand(time(0));};

  /** Main function: random generators act as functors, that return random numbers. 
	It´s non-const because it might modify a seed
	@return return a random number
	*/
  virtual T operator()() = 0;
  
  /** Return the class id. 
  @return the class name as a string
  */
  virtual string className() const { return "eoRandom"; };

  /**
   * Read object.
   * @param is A istream.
   * @throw runtime_exception If a valid object can't be read.
   */
  virtual void readFrom(istream& _is);
  
  /**
   * print object. Prints just the ID, since the seed is not accesible.
   * @param is A ostream.
   */
  void printOn(ostream& _os) const { _os << endl; };

private:
	/// true if the RNG has been started already. If it starts every time, means trouble.
	static bool started;
};

template<class T> bool eoRnd<T>::started = false;


//--------------------------------------------------------------------------

/**
 * Read object.
 * @param is A istream.
 * @throw runtime_exception If a valid object can't be read.
 */
template<class T>
void eoRnd<T>::readFrom(istream& _is) {
  if (!_is)   
    throw runtime_error("Problems reading from stream in eoRnd");
  long seed;
  _is >> seed;
  srand(seed);
}

#endif
