// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoHowMany_h.h
//   Base class for choosing a number of guys to apply something from a popsize
// (c) Marc Schoenauer, 2000
/* 
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

#ifndef eoHowMany_h
#define eoHowMany_h

/** A helper class, to determine a number of individuals from another one
 *  Typically, is used in selection / replacement procedures, e.g.
 *             the number of offspring from the number of parents, or
 *             the number of survivors for an eoReduce functor, ...
 * 
 * Such construct is very useful because in some cases you might not know the 
 * population size that will enter the replacement. For instance, you 
 * cannot simply have a pre-computed (double) rate of 1/popSize 
 * if you want to select or kill just 1 guy. Using an eoHowMany 
 * allows one to modify the population size without touching anything else.
 *
 * There are 4 possible way to compute the return value from the argument:
 *    - an absolute POSITIVE integer  --> return it (regardless of popsize)
 *    - a POSITIVE rate               -->  return rate*popSize
 *    - an absolute NEGATIVE integer  --> return popsize-rate (if positive)
 *    - a NEGATIVE rate in [-1,0]     --> store and use 1-|rate| (positive)
 * Note that a negative rate should be have been necessary because a rate is
 * relative, but it is there for consistency reasons - and because it
 * is needed in <a href="classeo_g3_replacement_h-source.html">eoG3Replacement</a>
 *
 * It has 2 private members, a double and an integer to cover all cases
 *
 * Example use: in <a href="class_eogeneralbreeder.html">eoGeneralBreeder.h</a>
 * Example reading from parser: in 
 *         <a href="make_algo_scalar_h-source.html">do/make_algo_scalar.h line 141</a>

 * MS 10/04/2002: 
 *    Added the possibility to have a negative number - 
 *        when treated as a number: returns then (size - combien)
 *    Should not modify anything when a positive number is passed in the ctor
 *
 * MS 20/06/2002:
 *    Added the negative rate and the operator-() (for
 *    eoG3Repalcement)
 *
 * It is an eoPersistent because we need to be able to use eoParamValue<eoHowMany>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream>
#endif

class eoHowMany : public eoPersistent
{
public:
  /** Original Ctor from direct rate + bool 
      @param rate    the rate, OR the integer to store, depending on 2nd arg.
      @param _interpret_as_rate to tell whether the rate actually is a rate
  */
  eoHowMany(double  _rate = 0.0, bool _interpret_as_rate = true):
    rate(_rate), combien(0)
  {
    if (_interpret_as_rate)
      {
	if (_rate<0)
	  {
	    rate = 1.0+_rate;
	    if (rate < 0)	   // was < -1
	      throw std::logic_error("rate<-1 in eoHowMany!");
	  }
      }
    else
      {
	rate = 0.0;		   // just in case, but shoud be unused
	combien = int(_rate);	   // negative values are allowed here
	if (combien != _rate)
	  std::cerr << "Warning: Number was rounded in eoHowMany";
      }
  }

  /** Ctor from an int - both from int and unsigned int are needed 
   *     to avoid ambiguity with the Ctor from a double */
  eoHowMany(int _combien) : rate(0.0), combien(_combien) {}

  /** Ctor from an unsigned int - both from int and unsigned int are needed 
   *     to avoid ambiguity with the Ctor from a double */
  eoHowMany(unsigned int _combien) : rate(0.0), combien(_combien) {}

  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoHowMany() {}

  /** Does what it was designed for 
   *  - combien==0 : return rate*_size
   *  - else
   *    - combien>0 : return combien (regardless of _size)
   *    - combien<0 : return _size-|combien|
   */
  unsigned int operator()(unsigned int _size)
  {
    if (combien == 0)
      {
	return (unsigned int) (rate * _size);
      }
    if (combien < 0)
      {
	unsigned int combloc = -combien;
	if (_size<combloc)
	  throw std::runtime_error("Negative result in eoHowMany");
	return _size-combloc;
      }
    return unsigned(combien);
  }

  virtual void printOn(std::ostream& _os) const 
  {
    if (combien == 0)
      _os << 100*rate << "% ";
    else
      _os << combien << " ";
    return;

  }

  virtual void readFrom(std::istream& _is) 
   {
    std::string value;
    _is >> value;
    readFrom(value);
    return;
  }

  void readFrom(std::string _value)
  {
    // check for %
    bool interpret_as_rate = false;   // == no %
    size_t pos =  _value.find('%');
    if (pos < _value.size())  //  found a %
      {
	interpret_as_rate = true;
	_value.resize(pos);	   // get rid of %
      }
    
#ifdef HAVE_SSTREAM
    std::istringstream is(_value);
#else
    std::istrstream is(_value.c_str());
#endif
    is >> rate;
    // now store
    if (interpret_as_rate)
      {
	combien = 0;
	rate /= 100.0;
      }
    else
      combien = int(rate);	   // and rate will not be used

    // minimal check
    if ( rate < 0.0 )
      throw std::runtime_error("Negative rate read in eoHowMany::readFrom");
  }

  /** The unary - operator: reverses the computation */
  eoHowMany operator-()
  {
    if (!combien)		   // only rate is used
      rate = 1.0-rate;
    else
      combien = -combien;
    return (*this);
  }
  
private :
  double rate;
  int combien;
};



#endif
