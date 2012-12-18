// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoHowMany_h.h
//   Base class for choosing a number of guys to apply something from a popsize
// (c) Marc Schoenauer, 2000
// (c) Thales group, 2010 (Johann Dréo <johann.dreo@thalesgroup.com>)

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

Contact: http://eodev.sourceforge.net

*/
//-----------------------------------------------------------------------------

#ifndef eoHowMany_h
#define eoHowMany_h

#include <sstream>

#include <utils/eoLogger.h>


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
 *        when treated as a number: returns then (size - count)
 *    Should not modify anything when a positive number is passed in the ctor
 *
 * MS 20/06/2002:
 *    Added the negative rate and the operator-() (for
 *    eoG3Repalcement)
 *
 * It is an eoPersistent because we need to be able to use eoParamValue<eoHowMany>
 *
 * @ingroup Core
 */
class eoHowMany : public eoPersistent
{
public:
  /** Original Ctor from direct rate + bool
      @param _rate    the rate, OR the integer to store, depending on 2nd arg.
      @param _interpret_as_rate to tell whether the rate actually is a rate
  */
  eoHowMany(double  _rate = 0.0, bool _interpret_as_rate = true):
    rate(_rate), count(0)
  {
    if (_interpret_as_rate)
      {
        if (_rate<0)
          {
            rate = 1.0+_rate;
            if (rate < 0)           // was < -1
              throw std::logic_error("rate<-1 in eoHowMany!");
          }
      }
    else
      {
        rate = 0.0;                   // just in case, but shoud be unused
        count = int(_rate);           // negative values are allowed here
        if (count != _rate)
          eo::log << eo::warnings << "Number was rounded in eoHowMany";
      }
  }

  /** Ctor from an int - both from int and unsigned int are needed
   *     to avoid ambiguity with the Ctor from a double */
  eoHowMany(int _count) : rate(0.0), count(_count) {}

  /** Ctor from an unsigned int - both from int and unsigned int are needed
   *     to avoid ambiguity with the Ctor from a double */
  eoHowMany(unsigned int _count) : rate(0.0), count(_count) {}

  /// Virtual dtor. They are needed in virtual class hierarchies.
  virtual ~eoHowMany() {}

  /** Does what it was designed for
   *  - count==0 : return rate*_size
   *  - else
   *    - count>0 : return count (regardless of _size)
   *    - count<0 : return _size-|count|
   */
  unsigned int operator()(unsigned int _size)
  {
    if (count == 0)
      {
        unsigned int res = static_cast<unsigned int>( std::ceil( rate * _size ) );

        if( res == 0 ) {
            eo::log << eo::warnings << "Call to a eoHowMany instance returns 0 (rate=" << rate << ", size=" << _size << ")" << std::endl;
        }

        return res;
      }
    if (count < 0)
      {
        unsigned int combloc = -count;
        if (_size<combloc)
          throw std::runtime_error("Negative result in eoHowMany");
        return _size-combloc;
      }
    return unsigned(count);
  }

  virtual void printOn(std::ostream& _os) const
  {
    if (count == 0)
      _os << 100*rate << "% ";
    else
      _os << count << " ";
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
        _value.resize(pos);           // get rid of %
      }

    std::istringstream is(_value);
    is >> rate;
    // now store
    if (interpret_as_rate)
      {
        count = 0;
        rate /= 100.0;
      }
    else
      count = int(rate);           // and rate will not be used

    // minimal check
    if ( rate < 0.0 )
      throw std::runtime_error("Negative rate read in eoHowMany::readFrom");
  }

  /** The unary - operator: reverses the computation */
  eoHowMany operator-()
  {
    if (!count)                   // only rate is used
      rate = 1.0-rate;
    else
      count = -count;
    return (*this);
  }

private :
  double rate;
  int count;
};



#endif
